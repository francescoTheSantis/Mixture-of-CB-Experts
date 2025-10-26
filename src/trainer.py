import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch
from torch.optim import AdamW
import numpy as np
import pandas as pd
from src.metrics import f1_acc_metrics
from tqdm import tqdm
#from models.l_cmr import LinearMemoryReasoner
from src.utils.scalers import StandardScaler

class Trainer:
    """
    Trainer class for the pytorch_lightning model.
    """
    def __init__(self, model, cfg, wandb_logger, csv_logger):
        self.cfg = cfg
        self.wandb_logger = wandb_logger
        self.csv_logger = csv_logger
        self.model = model
        self.epss = np.arange(0, 0.6, 0.1) # Noise levels for interventions
        self.p_ints = np.arange(0, 1.1, 0.1) # Intervention probabilities
        self.scale_target = cfg.scale_target if 'scale_target' in cfg else True

    def build_trainer(self):
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=self.cfg.patience, 
            verbose=True,
            mode='min'
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss', 
            filename='best_model', 
            save_top_k=1, 
            mode='min', 
            verbose=True
        )

        lr_monitor = LearningRateMonitor(logging_interval='step')

        loggers = [self.wandb_logger, self.csv_logger] if self.wandb_logger is not None else self.csv_logger

        self.trainer = pl.Trainer(
            max_epochs=self.cfg.max_epochs,
            callbacks=[early_stopping, checkpoint_callback, lr_monitor],
            logger=loggers,
            devices=self.cfg.gpus,  
            accelerator="auto",
            enable_progress_bar=True,
            gradient_clip_val=0.5
        )

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), 
                               lr=self.cfg.dataset.metadata.lr)

        LR_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                   mode='min', 
                                                                   factor=self.cfg.gamma, 
                                                                   patience=self.cfg.lr_patience, 
                                                                   verbose=True)
        self.scheduler = {
            'scheduler': LR_on_plateau,
            'monitor': 'val_loss',  
            'interval': 'epoch',
            'frequency': 1
        }

        # Set the optimizer in the respective model
        self.model.optimizer = self.optimizer
        self.model.scheduler = self.scheduler

    def train(self, train_dataloader, val_dataloader, ckpt_path=None):
        c_trues = []
        y_trues = []
        # iterate over the training-set
        for batch in train_dataloader:
            c = batch['c']
            y = batch['y']
            c_trues.append(c)
            y_trues.append(y)
        c_trues = torch.cat(c_trues, dim=0)
        y_trues = torch.cat(y_trues, dim=0)

        # If regression, standardize the target variable and store the scaler in the model
        if self.model.model.task == 'regression' and self.scale_target:
            # Fit scaler to y data
            scaler = StandardScaler(dims=(0,))
            scaler.fit(y_trues)

            # Store the scaler in the engine & model
            self.model.scaler = scaler
            self.model.model.scaler = scaler
        else:
            self.model.scaler = None
            self.model.model.scaler = None

        if self.model.model.__class__.__name__ == 'SymbolicMemoryReasoner':
            # setup the memory and selector
            self.model.model.setup_memory()
            # If KAN are used, we need to setup the grid for each kan in the memory.
            # This operation is required for any kind of task (classification, regression, ...).
            if self.model.model.equation_learning_strategy=='kan':
                self.model.model.setup_kan_grid(c_trues.to(self.cfg.gpus[0]))
                # Save c_true sin model as it will used to update the grid during training
                self.model.grid_inputs = c_trues.to(self.cfg.gpus[0])
                
        self.trainer.fit(self.model, 
                         train_dataloader, 
                         val_dataloader, ckpt_path=ckpt_path)

    def test(self, test_dataloader, ckpt_path=None):
        # Load the best model and test
        if ckpt_path is None:
            ckpt_path = self.trainer.checkpoint_callback.best_model_path
        self.trainer.test(self.model, test_dataloader, ckpt_path=ckpt_path)

    def fine_tune(self, 
                  train_dataloader, 
                  val_dataloader, 
                  log_dir='./'):

        print("\n" + "="*50)
        print("Get symbolic equation from KAN layers before fine-tuning")

        equations = []
        for layer in self.model.model.kan_layers:
            layer.auto_symbolic()
            equations.append(layer.symbolic_formula()[0][0])
        with open(f"{log_dir}/kan_equations_pre_fine_tuning.txt", "w") as f:
            for i, eq in enumerate(equations):
                f.write(f"KAN Layer {i+1}: {eq}\n")

        print("="*50)
        print("Starting Fine-tuning Phase")
        print("="*50)
        
        # Load the best checkpoint from initial training
        ckpt_path = self.trainer.checkpoint_callback.best_model_path
        
        if ckpt_path and ckpt_path != '':
            print(f"Loading checkpoint from: {ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint['state_dict'])
        
        # Set fine-tuning mode to change metric names
        self.model.fine_tuning = True
        self.model._set_metrics()
        
        fine_tune_lr = self.cfg.dataset.metadata.lr
        
        print(f"Fine-tuning learning rate: {fine_tune_lr}")
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = fine_tune_lr
        
        # Create new scheduler
        LR_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=self.cfg.gamma, 
            patience=self.cfg.lr_patience, 
            verbose=True
        )
        self.scheduler = {
            'scheduler': LR_on_plateau,
            'monitor': 'finetune/val_loss',  # Monitor fine-tuning val loss
            'interval': 'epoch',
            'frequency': 1
        }
        
        # Update optimizer and scheduler in model
        self.model.optimizer = self.optimizer
        self.model.scheduler = self.scheduler
        
        # Rebuild trainer with new configuration for fine-tuning
        early_stopping = EarlyStopping(
            monitor='finetune/val_loss',  # Monitor fine-tuning val loss
            patience=self.cfg.patience, 
            verbose=True,
            mode='min'
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='finetune/val_loss',  # Monitor fine-tuning val loss
            filename='best_model_finetuned', 
            save_top_k=1, 
            mode='min', 
            verbose=True
        )

        lr_monitor = LearningRateMonitor(logging_interval='step')

        loggers = [self.wandb_logger, self.csv_logger] if self.wandb_logger is not None else self.csv_logger

        self.trainer = pl.Trainer(
            max_epochs=self.cfg.max_epochs,
            callbacks=[early_stopping, checkpoint_callback, lr_monitor],
            logger=loggers,
            devices=self.cfg.gpus,  
            accelerator="auto",
            enable_progress_bar=True,
            gradient_clip_val=0.5
        )
        
        # Fine-tune
        self.trainer.fit(self.model, train_dataloader, val_dataloader)
        
        print("Fine-tuning completed!")
        print(f"Best fine-tuned model saved at: {self.trainer.checkpoint_callback.best_model_path}")
        
        equations = []
        for layer in self.model.model.kan_layers:
            equations.append(layer.symbolic_formula()[0][0])
        with open(f"{log_dir}/kan_equations_post_fine_tuning.txt", "w") as f:
            for i, eq in enumerate(equations):
                f.write(f"KAN Layer {i+1}: {eq}\n")        

    def interventions(self, test_dataloader, verbose=True):
        """
        Perform interventions on the test set and return the dataframe containing the results.
        Interventional accuracy is computed for different levels of noise and intervention probability.
        """
        # Pre-allocate list for better performance
        intervention_results = []
        
        # Set the model on the right device
        self.model = self.model.to(self.cfg.gpus[0])
        self.model.eval()
        self.model.model.test_interventions = True
        
        with torch.no_grad():
            for eps in self.epss:
                if verbose:
                    print('Performing interventions with noise:', eps)
                for p_int in tqdm(self.p_ints) if verbose else self.p_ints:
                    y_preds = []
                    y_trues = []
                    self.model.model.noise = eps
                    self.model.model.int_prob = p_int
                    
                    for batch in test_dataloader:
                        x, c, y = self.model.unpack_batch(batch)
                        
                        # Move the data to the GPU
                        if isinstance(x, dict):
                            x = {k: v.to(self.cfg.gpus[0]) for k, v in x.items()}
                        else:  
                            x = x.to(self.cfg.gpus[0])
                        c = c.to(self.cfg.gpus[0])
                        y = y.to(self.cfg.gpus[0])
                        
                        inputs = {'x': x, 'c': c, 'y': y}
                        output = self.model.forward(inputs)
                        output = self.model.model.filter_output_for_metrics(**output)
                        
                        # Move to CPU and detach to free GPU memory
                        y_pred = output[0].detach().cpu()
                        y_cpu = y.detach().cpu()
                        
                        y_preds.append(y_pred)
                        y_trues.append(y_cpu)
                        
                        # Clear GPU memory more aggressively
                        del x, c, y, inputs, output, y_pred, y_cpu
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    # Concatenate outside the loop
                    y = torch.cat(y_trues, dim=0).numpy()
                    y_preds = torch.cat(y_preds, dim=0)

                    if self.cfg.dataset.metadata.task == 'regression' and self.scale_target:
                        # If regression, inverse transform the predictions
                        y_preds = self.model.scaler.inverse_transform(y_preds)

                    y_preds = y_preds.numpy()

                    # Calculate metrics
                    if self.cfg.dataset.metadata.task == 'regression':
                        task_f1, task_acc = None, None
                        mse = np.mean((y - y_preds) ** 2)
                        mae = np.mean(np.abs(y - y_preds))
                        rmse = np.sqrt(mse)
                    else:
                        task_f1, task_acc = f1_acc_metrics(y, y_preds)
                        mse, mae, rmse = None, None, None

                    # Append to list instead of concatenating DataFrames
                    intervention_results.append({
                        'noise': round(eps, 1), 
                        'p_int': round(p_int, 1), 
                        'f1': task_f1, 
                        'accuracy': task_acc,
                        'mse': mse,
                        'mae': mae,
                        'rmse': rmse
                    })
                    
                    # Clear variables to free memory
                    del y, y_preds, y_trues
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        self.model.model.test_interventions = False
        
        # Create DataFrame once at the end
        intervention_df = pd.DataFrame(intervention_results)
        return intervention_df