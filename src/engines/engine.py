import os
from typing import Optional
from torch import nn
import torch
import pytorch_lightning as pl
from src.metrics import MAE, MSE, ClassAccuracy
from collections import OrderedDict
import pandas as pd
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection
import sympy as sp
import os
from tqdm import tqdm

from src.models.baselines.base import BaseModel
from src.utils.equation_storage import (
    clean_equation,
    parse_memory_equations,
    extract_per_sample_equations,
    extract_memory_equations
)

class Engine(pl.LightningModule):
    """
    PyTorch Lightning module wrapper.
    """
    def __init__(self,
                model: Optional[BaseModel] = None,
                c_names: Optional[list] = None,
                y_name: Optional[str] = None,
                csv_log_dir: Optional[str] = None,
                data_type: Optional[str] = None,
                dataset_name: Optional[str] = None,
                data_path: Optional[str] = None,
                scale_variables: bool = True,
                fine_tuning: bool = False,
                true_equations: Optional[list] = None
                ):
        super(Engine, self).__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"], logger=False)
        self.data_type = data_type
        self.c_names = c_names
        self.y_name = y_name
        self.num_classes = len(y_name) 
        self.class_names = y_name
        self.model_name = self.model.__class__.__name__

        self.csv_log_dir = csv_log_dir
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.scale_variables = scale_variables
        self.model.scale_variables = scale_variables
        self.fine_tuning = fine_tuning
        self.fine_tuning_stage = None  # Will be set to 'pruning' during fine-tuning after pruning
        self.true_equations = true_equations  # Store true equations if available

        # Initialize test predictions tracking
        self.test_predictions = []

        # Set the metrics
        self._set_metrics()

    @staticmethod
    def _check_metric(metric):
        metric = metric.clone()
        metric.reset()
        return metric

    def _set_metrics(self):
        # Add prefix for fine-tuning metrics
        if self.fine_tuning and self.fine_tuning_stage == 'allow_symbolic':
            prefix_modifier = "allow_symbolic/"
        elif self.fine_tuning and self.fine_tuning_stage == 'symbolic':
            prefix_modifier = "finetune_symbolic/"
        elif self.fine_tuning:
            prefix_modifier = "finetune/"
        else:
            prefix_modifier = ""
        
        if self.model.task == 'classification':
            self.train_y_metrics = MetricCollection(metrics={'acc': self._check_metric(ClassAccuracy())}, prefix=f"{prefix_modifier}train/y/")
            self.val_y_metrics = MetricCollection(metrics={'acc': self._check_metric(ClassAccuracy())}, prefix=f"{prefix_modifier}val/y/")
            self.test_y_metrics = MetricCollection(metrics={'acc': self._check_metric(ClassAccuracy())}, prefix="test/y/")
            self.train_c_metrics = MetricCollection(metrics={'acc': self._check_metric(ClassAccuracy())}, prefix=f"{prefix_modifier}train/c/")
            self.val_c_metrics = MetricCollection(metrics={'acc': self._check_metric(ClassAccuracy())}, prefix=f"{prefix_modifier}val/c/")
            self.test_c_metrics = MetricCollection(metrics={'acc': self._check_metric(ClassAccuracy())}, prefix="test/c/")
        elif self.model.task == 'regression':
            self.train_y_metrics = MetricCollection(metrics={'mse': self._check_metric(MSE()),
                                                             'mae': self._check_metric(MAE())}, prefix=f"{prefix_modifier}train/y/")
            self.val_y_metrics = MetricCollection(metrics={'mse': self._check_metric(MSE()),
                                                           'mae': self._check_metric(MAE())}, prefix=f"{prefix_modifier}val/y/")
            self.test_y_metrics = MetricCollection(metrics={'mse': self._check_metric(MSE()),
                                                            'mae': self._check_metric(MAE())}, prefix="test/y/")
            self.train_c_metrics = MetricCollection(metrics={'mse': self._check_metric(MSE()),
                                                             'mae': self._check_metric(MAE())}, prefix=f"{prefix_modifier}train/c/")
            self.val_c_metrics = MetricCollection(metrics={'mse': self._check_metric(MSE()),
                                                           'mae': self._check_metric(MAE())}, prefix=f"{prefix_modifier}val/c/")
            self.test_c_metrics = MetricCollection(metrics={'mse': self._check_metric(MSE()),
                                                            'mae': self._check_metric(MAE())}, prefix="test/c/")
        else:
            raise NotImplementedError(f"Metrics for task={self.model.task} not implemented.")
        
    def update_and_log_metrics(self, stage, y_hat, y, c_hat, c):
        # update and log task metrics
        y_collection = getattr(self, f"{stage}_y_metrics")
        y_collection.update(y_hat, y)
        self.log_dict(y_collection, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        # update and log concept metrics
        if self.model.has_concepts:
            c_collection = getattr(self, f"{stage}_c_metrics")
            c_collection.update(c_hat, c)
            self.log_dict(c_collection, on_step=False, on_epoch=True, logger=True, prog_bar=True)

    def forward(self, input):
        return self.model(input)

    def predict(self, input):
        return self.model(input)

    def unpack_batch(self, batch):
        x = batch['x']
        c = batch['c']
        y = batch['y']

        return x, c, y

    def shared_step(self, batch):
        # batch['x'] will be a tensor for image and toy datasets, and a dict for text datasets.

        # Maintain the shape of c to be (batch_size, n_concepts)
        batch['c'] = batch['c'] if batch['c'].ndim > 1 else batch['c'].unsqueeze(-1)

        # Scale concepts and targets BEFORE forward pass if needed
        if self.model.task == 'regression' and self.scale_variables:
            # Clone to avoid modifying the original batch data
            c_scaled = batch['c'].clone()
            y_scaled = batch['y'].clone()
            
            # Scale targets
            y_scaled = self.y_scaler.transform(y_scaled)
            
            # Scale concepts if model has them
            if self.model.has_concepts:
                for i, c_scaler in enumerate(self.c_scalers):
                    c_scaled[:, i:i+1] = c_scaler.transform(c_scaled[:, i:i+1])
            
            # Create a new batch dict with scaled values
            batch_scaled = {**batch, 'c': c_scaled, 'y': y_scaled}
        else:
            batch_scaled = batch

        # model forward (with scaled batch if scale_variables=True)
        model_output = self.forward(batch_scaled)

        # Compute loss
        y_hat_loss, c_hat_loss = self.model.filter_output_for_loss(**model_output)
        loss = self.model.loss(y_hat_loss, batch_scaled['y'], c_hat_loss, batch_scaled['c'])

        return loss, model_output

    def training_step(self, batch, batch_idx):
        self.model.global_step = self.current_epoch
        self.model.phase = 'train'
        loss, model_output = self.shared_step(batch)
        
        # Add prefix for fine-tuning
        if self.fine_tuning:
            loss_name = f"{self.fine_tuning_stage}/train_loss"
        else:
            loss_name = "train_loss"

        self.log(loss_name, loss.item(), on_step=False, on_epoch=True, logger=True, prog_bar=True)

        y_hat_metrics, c_hat_metrics = self.model.filter_output_for_metrics(**model_output)
        # compute task metrics
        # if the task is regression, we denormalize the predictions to compute the metrics on original scale
        if self.model.task == 'regression' and self.scale_variables:
            y_hat_metrics = self.y_scaler.inverse_transform(y_hat_metrics.detach())
            if self.model.has_concepts:
                c_hat_metrics_denorm = c_hat_metrics.clone().detach()
                for i, c_scaler in enumerate(self.c_scalers):
                    c_hat_metrics_denorm[:, i:i+1] = c_scaler.inverse_transform(c_hat_metrics[:, i:i+1].detach())
                c_hat_metrics = c_hat_metrics_denorm
        # Use original unscaled batch for ground truth metrics
        self.update_and_log_metrics('train', y_hat_metrics, batch['y'], c_hat_metrics, batch['c'])

        return loss

    def validation_step(self, batch, batch_idx):
        self.model.phase = 'val'
        loss, model_output = self.shared_step(batch)
        
        # Add prefix for fine-tuning
        if self.fine_tuning:
            loss_name = f"{self.fine_tuning_stage}/val_loss"
        else:
            loss_name = "val_loss"

        self.log(loss_name, loss.item(), on_step=False, on_epoch=True, logger=True, prog_bar=True)

        y_hat_metrics, c_hat_metrics = self.model.filter_output_for_metrics(**model_output)
        # compute task metrics
        # if the task is regression, we denormalize the predictions to compute the metrics on original scale
        if self.model.task == 'regression' and self.scale_variables:
            y_hat_metrics = self.y_scaler.inverse_transform(y_hat_metrics)
            if self.model.has_concepts:
                c_hat_metrics_denorm = c_hat_metrics.clone()
                for i, c_scaler in enumerate(self.c_scalers):
                    c_hat_metrics_denorm[:, i:i+1] = c_scaler.inverse_transform(c_hat_metrics[:, i:i+1])
                c_hat_metrics = c_hat_metrics_denorm
        # Use original unscaled batch for ground truth metrics
        self.update_and_log_metrics('val', y_hat_metrics, batch['y'], c_hat_metrics, batch['c'])

        return loss 

    def on_test_start(self):
        """Called at the start of testing. Cache expensive equation extractions."""
        print("\n[Test Setup] Caching equations for fast storage...")
        # Cache equations for models that use get_symbolic_equivalent
        if self.model_name in ['BlackBox', 'ConceptEmbeddingModel']:
            # For BlackBox and ConceptEmbeddingModel, store NaN instead of extracting equations
            self.cached_equations = {0: "NaN"}
            self.cached_parsed_equations = {(0, y_name): "NaN" for y_name in self.y_name}
            print(f"✓ Set equations to NaN for {self.model_name}")
        elif self.model_name == 'ConceptBottleneckModel':
            try:
                with tqdm(total=2, desc="Extracting equations", leave=False) as pbar:
                    pbar.set_description("Extracting symbolic equations")
                    eq_result = self.model.get_symbolic_equivalent(return_equations=True)
                    pbar.update(1)
                    
                    # Store cached equations for reuse
                    # Use hasattr to check if it's iterable instead of isinstance to avoid potential issues
                    try:
                        # Try to iterate - if it's a list/tuple, this will work
                        eq_strs = [f"{self.y_name[i]}: {str(eq)}" for i, eq in enumerate(eq_result)]
                        self.cached_equations = {0: "; ".join(eq_strs)}
                    except (TypeError, AttributeError):
                        # Single output - not iterable
                        self.cached_equations = {0: f"{self.y_name[0]}: {str(eq_result)}"}
                    
                    pbar.set_description("Parsing equations")
                    # Parse the equations for fast lookup
                    self.cached_parsed_equations = parse_memory_equations(self.cached_equations, self.y_name)
                    pbar.update(1)
                print(f"✓ Cached {len(self.cached_parsed_equations)} equation(s)")
            except Exception as e:
                print(f"✗ Error extracting equations: {str(e)}")
                self.cached_equations = {0: f"Error extracting equation: {str(e)}"}
                self.cached_parsed_equations = {(0, y_name): f"Error extracting equation: {str(e)}" for y_name in self.y_name}
        elif self.model_name in ['KANSymbolicCBM', 'LinearSymbolicCBM', 'PriorSymbolicCBM', 'SymbolicRegressorCBM', 'MemoryCBM']:
            # For memory-based models, extract and parse equations once
            with tqdm(total=2, desc="Processing memory equations", leave=False) as pbar:
                pbar.set_description("Extracting memory equations")
                self.cached_equations = extract_memory_equations(self.model, self.model_name)
                pbar.update(1)
                
                pbar.set_description("Parsing memory equations")
                self.cached_parsed_equations = parse_memory_equations(self.cached_equations, self.y_name)
                pbar.update(1)
            print(f"✓ Cached {len(self.cached_parsed_equations)} memory equation(s)")
        else:
            self.cached_equations = None
            self.cached_parsed_equations = None
    
    def test_step(self, batch, batch_idx):
        self.model.phase = 'test'
        loss, model_output = self.shared_step(batch)
        self.log("test_loss", loss.item(), on_step=False, on_epoch=True, logger=True, prog_bar=True)

        y_hat_metrics, c_hat_metrics = self.model.filter_output_for_metrics(**model_output)
        # compute task metrics
        # if the task is regression, we denormalize the predictions to compute the metrics on original scale
        if self.model.task == 'regression' and self.scale_variables:
            y_hat_metrics = self.y_scaler.inverse_transform(y_hat_metrics)
            if self.model.has_concepts:
                c_hat_metrics_denorm = c_hat_metrics.clone()
                for i, c_scaler in enumerate(self.c_scalers):
                    c_hat_metrics_denorm[:, i:i+1] = c_scaler.inverse_transform(c_hat_metrics[:, i:i+1])
                c_hat_metrics = c_hat_metrics_denorm
        # Use original unscaled batch for ground truth metrics
        self.update_and_log_metrics('test', y_hat_metrics, batch['y'], c_hat_metrics, batch['c'])

        # Collect per-sample predictions for analysis
        self._collect_test_sample_data(batch, batch_idx, model_output, y_hat_metrics, c_hat_metrics)

        return loss 
    
    def _denormalize(self, tensor):
        if self.model.task == 'regression':
            return tensor * self.model.y_std + self.model.y_mean
        else:
            return tensor

    def on_train_epoch_end(self):
        if self.model_name == 'KANSymbolicCBM' and not self.model.symbolic_predictors:
            # Update the KAN grid (self.grid_inputs is already scaled from trainer)
            if self.current_epoch % 10 == 0 :
                self.model.setup_kan_grid(self.grid_inputs)

    def _collect_test_sample_data(self, batch, batch_idx, model_output, y_hat_metrics, c_hat_metrics):
        """
        Collect per-sample data during testing for later analysis.
        """
        batch_size = batch['y'].shape[0]
        
        # Get the selected memory slot index for each sample
        if 'selection_dist' in model_output:
            # With independent outputs: Shape is (batch_size, n_outputs, memory_size)
            selection_dist = model_output['selection_dist']
            # Independent outputs: get argmax for each output
            # Shape: (batch_size, n_outputs)
            selected_memory = torch.argmax(selection_dist, dim=1)

        else:
            # No memory selection available (for BlackBox and ConceptEmbeddingModel)
            selected_memory = torch.zeros(batch_size, dtype=torch.long)
        
        # Batch convert all tensors to numpy once (optimization)
        predictions_np = y_hat_metrics.detach().cpu().numpy()
        y_true_np = batch['y'].detach().cpu().numpy()
        c_true_np = batch['c'].detach().cpu().numpy()
        c_pred_np = c_hat_metrics.detach().cpu().numpy() if c_hat_metrics is not None else None
        
        # Extract equations for this batch
        if self.model_name == 'LinearConceptEmbeddingModel':
            # For LinearConceptEmbeddingModel, equations are per-sample (not per-memory-slot)
            # Pass predictions to optimize equation extraction (only extract needed equations)
            equations_per_sample = extract_per_sample_equations(
                model_output, batch_size, predictions_np, self.c_names, self.y_name, self.model.task
            )
        elif self.model_name in ['DeepConceptReasoner', 'ConceptMemoryReasoner']:
            # For DeepConceptReasoner and ConceptMemoryReasoner, equations are per-sample boolean rules
            with torch.no_grad():
                equations_per_sample = self.model.get_local_explanations(batch['x'])
        else:
            # For other models, use cached parsed equations if available
            if hasattr(self, 'cached_parsed_equations') and self.cached_parsed_equations is not None:
                parsed_equations = self.cached_parsed_equations
            else:
                # Fallback: extract and parse equations on-the-fly
                equations_per_slot = extract_memory_equations(self.model, self.model_name)
                parsed_equations = parse_memory_equations(equations_per_slot, self.y_name)
        
        # Store data for each sample in the batch
        for i in range(batch_size):
            prediction = predictions_np[i]
            y_true = y_true_np[i]
        
            # Get equation based on model type
            if self.model_name in ['LinearConceptEmbeddingModel']:
                # Per-sample equations (already formatted)
                equation = equations_per_sample[i]
            elif self.model_name in ['DeepConceptReasoner', 'ConceptMemoryReasoner']:
                # Per-sample boolean rule explanations
                equation = list(equations_per_sample[i].values())[0]
            else:
                # Single-output task: store only the equation for the predicted class
                if self.model.task == 'classification':
                    # For classification, prediction is the predicted class index
                    pred_class_idx = int(prediction) if prediction.ndim == 0 else int(prediction[0])
                else:
                    # For regression, we have a single output
                    pred_class_idx = 0
                
                y_name = self.y_name[pred_class_idx] if len(self.y_name) > 1 else self.y_name[0]
                
                if self.model_name in ['BlackBox', 'ConceptEmbeddingModel', 'ConceptBottleneckModel']:
                    memory_idx = 0  # No memory slots, use default
                else:
                    memory_idx = selected_memory[i, pred_class_idx].item() if (len(self.y_name) > 1 and self.model.task == 'classification') else selected_memory[i].item()
                
                # Use pre-parsed equations for fast lookup
                equation = parsed_equations.get((memory_idx, y_name), "N/A")

            c_pred = c_pred_np[i] if c_pred_np is not None else None

            sample_data = {
                'sample_idx': batch_idx * batch_size + i,
                'equation': clean_equation(equation),
                'c_pred': c_pred,
                'y_pred': prediction,
                'c_true': c_true_np[i],
                'y_true': y_true,
            }
            
            self.test_predictions.append(sample_data)

    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch. Saves per-sample predictions to CSV.
        """
        if not hasattr(self, 'test_predictions') or len(self.test_predictions) == 0:
            return
        
        print(f"\nProcessing {len(self.test_predictions)} test samples...")
        
        # Convert to DataFrame for easy saving
        # Flatten arrays for CSV storage
        records = []
        for pred in self.test_predictions:
            record = {
                'sample_idx': pred['sample_idx'],
                'equation': pred['equation'],
            }
            # Add true equation if available
            if 'true_equation' in pred:
                record['true_equation'] = pred['true_equation']
            
            # Add task predictions and ground truth
            y_true = pred['y_true']
            y_pred = pred['y_pred']
            
            # Handle both single and multi-output tasks
            if y_true.ndim == 0 or (y_true.ndim == 1 and len(y_true) == 1):
                # Single output
                y_true_val = float(y_true) if y_true.ndim == 0 else float(y_true[0])
                y_pred_val = float(y_pred) if y_pred.ndim == 0 else float(y_pred[0])
                
                record['y_true'] = y_true_val
                record['y_pred'] = y_pred_val
                
                # Add task names for classification
                if self.model.task == 'classification':
                    pred_class_idx = int(y_pred_val)
                    true_class_idx = int(y_true_val)
                    record['y_pred_task_name'] = self.class_names[pred_class_idx] if pred_class_idx < len(self.class_names) else f"class_{pred_class_idx}"
                    record['y_true_task_name'] = self.class_names[true_class_idx] if true_class_idx < len(self.class_names) else f"class_{true_class_idx}"
                else:
                    # For regression, use the single task name
                    task_name = self.class_names[0] if len(self.class_names) == 1 else "output"
                    record['y_pred_task_name'] = task_name
                    record['y_true_task_name'] = task_name
            else:
                # Multi-output
                for task_idx, task_name in enumerate(self.class_names):
                    record[f'y_true_{task_name}'] = float(y_true[task_idx])
                    record[f'y_pred_{task_name}'] = float(y_pred[task_idx])
                
                # For multi-output, store all task names (not applicable for single prediction/true value)
                record['y_pred_task_name'] = "; ".join(self.class_names)
                record['y_true_task_name'] = "; ".join(self.class_names)
            
            # Add concept columns
            c_true = pred['c_true']
            c_pred = pred['c_pred']
            
            for c_idx, c_name in enumerate(self.c_names):
                record[f'c_true_{c_name}'] = float(c_true[c_idx]) if c_true.ndim > 0 else float(c_true)
                if c_pred is not None:
                    record[f'c_pred_{c_name}'] = float(c_pred[c_idx]) if c_pred.ndim > 0 else float(c_pred)
                else:
                    record[f'c_pred_{c_name}'] = None
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Save to CSV in the log directory
        save_path = os.path.join(self.csv_log_dir, 'test_predictions_per_sample.csv')
        df.to_csv(save_path, index=False)
        print(f"✓ Saved test predictions to: {save_path}")
        print(f"  Total samples: {len(records)}")
        print(f"  Columns: {list(df.columns)}")
        
        # Clear the predictions list for potential future test runs
        self.test_predictions = []

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]