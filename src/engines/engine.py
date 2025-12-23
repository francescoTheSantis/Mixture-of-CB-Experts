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

from src.models.baselines.base import BaseModel

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

        self.log(loss_name, loss.item())

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

        self.log(loss_name, loss.item())

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
        # Cache equations for models that use get_symbolic_equivalent
        if self.model_name in ['BlackBox', 'ConceptEmbeddingModel']:
            try:
                eq_result = self.model.get_symbolic_equivalent(return_equations=True)
                # Store cached equations for reuse
                # Use hasattr to check if it's iterable instead of isinstance to avoid potential issues
                try:
                    # Try to iterate - if it's a list/tuple, this will work
                    eq_strs = [f"{self.y_name[i]}: {str(eq)}" for i, eq in enumerate(eq_result)]
                    self.cached_equations = {0: "; ".join(eq_strs)}
                except (TypeError, AttributeError):
                    # Single output - not iterable
                    self.cached_equations = {0: f"{self.y_name[0] if isinstance(self.y_name, list) else self.y_name}: {str(eq_result)}"}
            except Exception as e:
                self.cached_equations = {0: f"Error extracting equation: {str(e)}"}
        elif self.model_name in ['KANSymbolicCBM', 'LinearSymbolicCBM', 'PriorSymbolicCBM', 'SymbolicRegressorCBM']:
            # For memory-based models, extract and parse equations once
            self.cached_equations = self._extract_memory_equations()
            self.cached_parsed_equations = self._parse_memory_equations(self.cached_equations)
        else:
            self.cached_equations = None
            self.cached_parsed_equations = None
    
    def test_step(self, batch, batch_idx):
        self.model.phase = 'test'
        loss, model_output = self.shared_step(batch)
        self.log("test_loss", loss.item())

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
        if self.model_name in ['KANSymbolicCBM', 'LinearSymbolicCBM', 'PriorSymbolicCBM', 'SymbolicRegressorCBM', \
                               'BlackBox', 'ConceptEmbeddingModel', 'LinearConceptEmbeddingModel', 'DeepConceptReasoner', 'ConceptMemoryReasoner']:
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
            equations_per_sample = self._extract_per_sample_equations(model_output, batch_size)
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
                equations_per_slot = self._extract_memory_equations()
                parsed_equations = self._parse_memory_equations(equations_per_slot)
        
        # Store data for each sample in the batch
        for i in range(batch_size):
            prediction = predictions_np[i]
            y_true = y_true_np[i]
            
            # Determine if this is a single-output or multi-output task
            is_multi_output = y_true.ndim > 0 and len(y_true) > 1
            
            # Get equation based on model type
            if self.model_name in ['LinearConceptEmbeddingModel']:
                # Per-sample equations (already formatted)
                equation = equations_per_sample[i]
            elif self.model_name in ['DeepConceptReasoner', 'ConceptMemoryReasoner']:
                # Per-sample boolean rule explanations
                equation = list(equations_per_sample[i].values())[0]
            else:
                # Memory-based models with independent selection per output
                if is_multi_output:
                    # Multi-output task: store equations for all outputs
                    eq_parts = []
                    for out_idx, y_name in enumerate(self.y_name):
                        memory_idx = selected_memory[i, out_idx].item()
                        # Use pre-parsed equations for fast lookup
                        eq_parts.append(parsed_equations.get((memory_idx, y_name), "N/A"))
                    equation = "; ".join(eq_parts)
                else:
                    # Single-output task: store only the equation for the predicted class
                    if self.model.task == 'classification':
                        # For classification, prediction is the predicted class index
                        pred_class_idx = int(prediction) if prediction.ndim == 0 else int(prediction[0])
                    else:
                        # For regression, we have a single output
                        pred_class_idx = 0
                    
                    y_name = self.y_name[pred_class_idx] if len(self.y_name) > 1 else self.y_name[0]
                    
                    if self.model_name in ['BlackBox', 'ConceptEmbeddingModel']:
                        memory_idx = 0  # No memory slots, use default
                    else:
                        memory_idx = selected_memory[i, pred_class_idx].item() if (len(self.y_name) > 1 and self.model.task == 'classification') else selected_memory[i].item()
                    
                    # Use pre-parsed equations for fast lookup
                    equation = parsed_equations.get((memory_idx, y_name), "N/A")

            c_pred = c_pred_np[i] if c_pred_np is not None else None

            sample_data = {
                'sample_idx': batch_idx * batch_size + i,
                'equation': equation,
                'c_pred': c_pred,
                'y_pred': prediction,
                'c_true': c_true_np[i],
                'y_true': y_true,
            }
            
            self.test_predictions.append(sample_data)

    def _parse_memory_equations(self, equations_per_slot):
        """
        Pre-parse equation strings into a structured format for fast lookup.
        Returns a dictionary mapping (memory_idx, output_name) -> equation_string.
        This avoids repeated string parsing inside the per-sample loop.
        """
        parsed = {}
        
        for mem_idx, slot_equations in equations_per_slot.items():
            if ';' in slot_equations:
                # Multi-output format: "y0: eq0; y1: eq1; ..."
                eqs = slot_equations.split(';')
                for eq in eqs:
                    eq = eq.strip()
                    if ':' in eq:
                        # Extract output name and equation
                        output_name, equation = eq.split(':', 1)
                        parsed[(mem_idx, output_name.strip())] = equation.strip()
            else:
                # Single output format
                if ':' in slot_equations:
                    output_name, equation = slot_equations.split(':', 1)
                    parsed[(mem_idx, output_name.strip())] = equation.strip()
                else:
                    # No prefix - assume it applies to all outputs
                    for y_name in self.y_name:
                        parsed[(mem_idx, y_name)] = slot_equations.strip()
        
        return parsed

    def _extract_per_sample_equations(self, model_output, batch_size):
        """
        Extract equation strings for each sample (used by LinearConceptEmbeddingModel).
        Returns a list of equation strings, one per sample.
        """
        equations = []
        
        # Extract weights and bias from model output
        # weights shape: (batch_size, 1, n_concepts, n_outputs)
        weights = model_output['weights'].detach().cpu().numpy()
        
        # y_bias shape: (batch_size, 1, n_outputs) if present, else None
        y_bias = model_output.get('y_bias', None)
        if y_bias is not None:
            y_bias = y_bias.detach().cpu().numpy()
        
        # Build equation for each sample
        for sample_idx in range(batch_size):
            eq_strs = []
            n_outputs = weights.shape[-1]
            n_concepts = weights.shape[-2]
            
            for out_idx in range(n_outputs):
                terms = []
                # Add weighted concept terms
                for c_idx in range(n_concepts):
                    weight = weights[sample_idx, 0, c_idx, out_idx]
                    if abs(weight) > 1e-6:  # Only include non-zero terms
                        c_name = self.c_names[c_idx]
                        terms.append(f"{weight:.4f}*{c_name}")
                
                # Add bias if present
                if y_bias is not None:
                    bias_value = y_bias[sample_idx, 0, out_idx]
                    terms.append(f"{bias_value:.4f}")
                
                # Build equation string
                y_name = self.y_name[out_idx] if len(self.y_name) > 1 else self.y_name[0]
                eq_str = f"{y_name}: " + " + ".join(terms) if terms else f"{y_name}: 0"
                eq_strs.append(eq_str)
            
            equations.append("; ".join(eq_strs))
        
        return equations

    def _extract_memory_equations(self):
        """
        Extract equation strings for each memory slot based on model type.
        Returns a dictionary mapping memory_slot_idx -> equation_string.
        """
        equations = {}
        model_name = self.model_name
        
        if model_name == 'KANSymbolicCBM':
            # Extract from KAN predictor or SymbolicPredictor
            if hasattr(self.model.predictor, 'trainable_equations'):
                # SymbolicPredictor with learned equations
                for mem_idx, set_name in enumerate(sorted(self.model.predictor.trainable_equations.keys())):
                    eq_strs = []
                    for eq_name in self.model.predictor.equation_names[set_name]:
                        eq_module = self.model.predictor.trainable_equations[set_name][eq_name]
                        eq_strs.append(f"{eq_name}: {eq_module.get_equation_string()}")
                    equations[mem_idx] = "; ".join(eq_strs)
            elif hasattr(self.model.predictor, 'kans'):
                # KANPredictor - abstract representation
                for mem_idx in range(len(self.model.predictor.kans)):
                    equations[mem_idx] = f"KAN{mem_idx}[{self.model.widths}]"
            else:
                equations[0] = "No equations available"
        
        elif model_name == 'LinearSymbolicCBM':
            # Extract linear equations from memory
            try:
                equation_weights = self.model.linear_memory_predictor.equation_decoder(
                    self.model.linear_memory_predictor.equation_memory.weight
                )
                equation_weights = equation_weights.view(
                    self.model.memory_size, 
                    len(self.model.linear_memory_predictor.parameters), 
                    len(self.model.y_names)
                )
                weights_np = equation_weights.detach().cpu().numpy()
                
                for mem_idx in range(self.model.memory_size):
                    eq_strs = []
                    for out_idx, y_name in enumerate(self.model.y_names):
                        # Build equation string
                        terms = []
                        n_concepts = len(self.model.c_names)
                        for c_idx in range(n_concepts):
                            weight = weights_np[mem_idx, c_idx, out_idx]
                            if abs(weight) > 1e-6:  # Only include non-zero terms
                                terms.append(f"{weight:.4f}*{self.model.c_names[c_idx]}")
                        
                        # Add bias
                        if self.model.bias == 'local':
                            bias_value = weights_np[mem_idx, -1, out_idx]
                            terms.append(f"{bias_value:.4f}")
                        elif self.model.bias == 'global':
                            bias_value = self.model.linear_memory_predictor.bias_params[out_idx].item()
                            terms.append(f"{bias_value:.4f}")
                        
                        eq_str = f"{y_name}: " + " + ".join(terms) if terms else f"{y_name}: 0"
                        eq_strs.append(eq_str)
                    
                    equations[mem_idx] = "; ".join(eq_strs)
            except Exception as e:
                for mem_idx in range(getattr(self.model, 'memory_size', 1)):
                    equations[mem_idx] = f"Error extracting equation: {str(e)}"
        
        elif model_name == 'PriorSymbolicCBM':
            # Extract from prior_predictor
            if hasattr(self.model.prior_predictor, 'trainable_equations'):
                for mem_idx, set_name in enumerate(sorted(self.model.prior_predictor.trainable_equations.keys())):
                    eq_strs = []
                    for eq_name in self.model.prior_predictor.equation_names[set_name]:
                        eq_module = self.model.prior_predictor.trainable_equations[set_name][eq_name]
                        eq_strs.append(f"{eq_name}: {eq_module.get_equation_string()}")
                    equations[mem_idx] = "; ".join(eq_strs)
            else:
                equations[0] = "No equations available"
        
        elif model_name == 'SymbolicRegressorCBM':
            # Extract from predictor
            if hasattr(self.model.predictor, 'trainable_equations'):
                # SymbolicPredictor with learned equations
                for mem_idx, set_name in enumerate(sorted(self.model.predictor.trainable_equations.keys())):
                    eq_strs = []
                    for eq_name in self.model.predictor.equation_names[set_name]:
                        eq_module = self.model.predictor.trainable_equations[set_name][eq_name]
                        # NOTE: get_equation_string() returns equations with current fine-tuned parameter values
                        eq_strs.append(f"{eq_name}: {eq_module.get_equation_string()}")
                    equations[mem_idx] = "; ".join(eq_strs)
            else:
                # BlackBoxPredictor or not yet trained
                for mem_idx in range(getattr(self.model, 'memory_size', 1)):
                    equations[mem_idx] = "No symbolic equations (BlackBoxPredictor)"
        
        elif model_name == 'BlackBox':
            # Use cached equations if available (set in on_test_start)
            if hasattr(self, 'cached_equations') and self.cached_equations is not None:
                equations = self.cached_equations
            else:
                # Fallback: Extract equation from predictor using get_symbolic_equivalent
                try:
                    eq_result = self.model.get_symbolic_equivalent(return_equations=True)
                    # For multi-output, eq_result is a list of equations
                    if isinstance(eq_result, list):
                        eq_strs = [f"{self.y_name[i]}: {str(eq)}" for i, eq in enumerate(eq_result)]
                        equations[0] = "; ".join(eq_strs)
                    else:
                        # Single output
                        equations[0] = f"{self.y_name[0] if isinstance(self.y_name, list) else self.y_name}: {str(eq_result)}"
                except Exception as e:
                    equations[0] = f"Error extracting equation: {str(e)}"
        
        elif model_name == 'ConceptEmbeddingModel':
            # Use cached equations if available (set in on_test_start)
            if hasattr(self, 'cached_equations') and self.cached_equations is not None:
                equations = self.cached_equations
            else:
                # Fallback: Extract equation from y_predictor using get_symbolic_equivalent
                try:
                    eq_result = self.model.get_symbolic_equivalent(return_equations=True)
                    # For multi-output, eq_result is a list of equations
                    if isinstance(eq_result, list):
                        eq_strs = [f"{self.y_name[i]}: {str(eq)}" for i, eq in enumerate(eq_result)]
                        equations[0] = "; ".join(eq_strs)
                    else:
                        # Single output
                        equations[0] = f"{self.y_name[0] if isinstance(self.y_name, list) else self.y_name}: {str(eq_result)}"
                except Exception as e:
                    equations[0] = f"Error extracting equation: {str(e)}"
        
        else:
            # Other models - no memory-based equations
            equations[0] = f"Model {model_name} does not use memory-based equations"
        
        return equations

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
        print(f"✓ Saved per-sample test predictions to: {save_path}")
        print(f"  Total samples: {len(records)}")
        print(f"  Columns: {list(df.columns)}")
        
        # Clear the predictions list for potential future test runs
        self.test_predictions = []

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]