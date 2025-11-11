import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch
from torch.optim import AdamW
import numpy as np
import pandas as pd
from src.metrics import f1_acc_metrics
from tqdm import tqdm
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

        # Store checkpoint directory to ensure it's consistent across all phases
        self.checkpoint_dir = self.csv_logger.log_dir
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            monitor='val_loss', 
            filename='best_model', 
            save_top_k=1, 
            mode='min', 
            verbose=True,
            save_last=False,
            enable_version_counter=False  # Prevent version suffixes
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
            # gradient_clip_val=0.5
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

        if self.model.model.__class__.__name__ == 'KANSymbolicCBM':
            self.kan_inputs = c_trues.to(self.cfg.gpus[0])
            self.model.model.setup_kan_grid(self.kan_inputs)
            # Save c_trues in model as it will used to update the grid during training
            self.model.grid_inputs = c_trues.to(self.cfg.gpus[0])

        self.trainer.fit(self.model, 
                         train_dataloader, 
                         val_dataloader, ckpt_path=ckpt_path)

    def test(self, test_dataloader, ckpt_path=None):
        # Load the best model and test
        if ckpt_path is None:
            ckpt_path = f"{self.checkpoint_dir}/best_model.ckpt"
        self.trainer.test(self.model, test_dataloader, ckpt_path=ckpt_path)

    def allow_symbolic(self, train_dataloader, val_dataloader):
        """
        Allow symbolic execution for the model.
        Train the model for a few epochs to store the activation functions in order to allow
        symbolic substitution of the splines.
        """

        model_name = self.cfg.model.metadata.name
        # The kan model needs this training to just process the entire training set once and store the activations.
        # therefore just 1 epoch is needed.
        epochs = 1 if model_name == 'kan_symbolic_cbm' else self.cfg.max_epochs

        # Load the best checkpoint from initial training (best_model.ckpt)
        ckpt_path = f"{self.checkpoint_dir}/best_model.ckpt"
        
        print(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        print("\n" + "="*50)
        print("Allowing Symbolic substitution")
        print("="*50)
        
        # Allow Symbolic substitution for the KAN layers
        if model_name == 'kan_symbolic_cbm':
            self.model.model.allow_symbolic()

            # NOTE: if you want, you can prune the KAN layers before allowing symbolic execution.
            # Unfortunatelly, the pruning does not work when speed_up_training=True.
            # So, if you want to prune, set speed_up_training=False in the model config.
            # self.model.model.prune()

            # Update the grid
            self.model.model.setup_kan_grid(self.kan_inputs)
            
        # For SR-Sym-CBM, collect data for symbolic fine-tuning
        elif model_name == 'sr_symbolic_cbm':
            self.model.eval()
            self.model = self.model.to(self.cfg.gpus[0])
            # Reset stored tensors before collecting
            self.model.model.reset_stored_tensors()
            with torch.no_grad():
                for batch in tqdm(train_dataloader, desc="Storing training data"):
                    x, c, y = self.model.unpack_batch(batch)
                    # Move the data to the GPU
                    if isinstance(x, dict):
                        x = {k: v.to(self.cfg.gpus[0]) for k, v in x.items()}
                    else:  
                        x = x.to(self.cfg.gpus[0])
                    c = c.to(self.cfg.gpus[0])
                    y = y.to(self.cfg.gpus[0])
                    inputs = {'x': x, 'c': c, 'y': y}
                    # Forward pass with storage enabled
                    _ = self.model.model.forward(inputs, store_for_finetuning=True)
                    # Clear GPU memory
                    del x, c, y, inputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            self.model = self.model.to('cpu')
            # Run symbolic fine-tuning
            self.model.model.run_symbolic_finetuning()
    
        # Set fine-tuning mode to change metric names
        self.model.fine_tuning = True
        self.model.fine_tuning_stage = 'allow_symbolic'
        self.model._set_metrics()
        
        fine_tune_lr = self.cfg.dataset.metadata.lr
                
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
            'monitor': 'allow_symbolic/val_loss',  # Monitor fine-tuning val loss 
            'interval': 'epoch',
            'frequency': 1
        }
        
        # Update optimizer and scheduler in model
        self.model.optimizer = self.optimizer
        self.model.scheduler = self.scheduler
        
        # Rebuild trainer with new configuration for fine-tuning
        early_stopping = EarlyStopping(
            monitor='allow_symbolic/val_loss',  # Monitor fine-tuning val loss 
            patience=self.cfg.patience, 
            verbose=True,
            mode='min'
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            monitor='allow_symbolic/val_loss',  # Monitor fine-tuning val loss 
            filename='best_model', 
            save_top_k=1, 
            mode='min', 
            verbose=True,
            save_last=False,
            enable_version_counter=False  # Prevent version suffixes
        )

        lr_monitor = LearningRateMonitor(logging_interval='step')

        loggers = [self.wandb_logger, self.csv_logger] if self.wandb_logger is not None else self.csv_logger

        self.trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=[early_stopping, checkpoint_callback, lr_monitor],
            logger=loggers,
            devices=self.cfg.gpus,  
            accelerator="auto",
            enable_progress_bar=True,
            # gradient_clip_val=0.5
        )
        
        # Fine-tune after pruning
        self.trainer.fit(self.model, train_dataloader, val_dataloader)
        
        return f"{self.checkpoint_dir}/best_model.ckpt"

    def fine_tune(self, 
                  train_dataloader, 
                  val_dataloader, 
                  log_dir='./',
                  ckpt_path=None):
        """
        Fine-tune the model with symbolic expressions replacing KAN layers.
        This is the second phase of fine-tuning for KAN-based models.
        """

        # Load the best checkpoint from pruning phase (best_model.ckpt)
        ckpt_path = f"{self.checkpoint_dir}/best_model.ckpt"
        
        if ckpt_path and ckpt_path != '':
            print(f"Loading checkpoint from: {ckpt_path}")
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint['state_dict'])
        
        print("\n" + "="*50)
        print("Get symbolic equation from KAN layers before fine-tuning")

        self.model.model.get_learned_equations(log_dir)

        # Update the grid after symbolic conversion
        #self.model.model.setup_kan_grid(self.kan_inputs)

        print("="*50)
        print("Starting Fine-tuning Phase (Symbolic)")
        print("="*50)
    
        # Set fine-tuning mode to change metric names
        self.model.fine_tuning = True
        self.model.fine_tuning_stage = 'symbolic'
        self.model._set_metrics()
        
        fine_tune_lr = self.cfg.dataset.metadata.lr
        
        print(f"Fine-tuning learning rate: {fine_tune_lr}")
        
        # Recreate optimizer with only trainable parameters
        # This is crucial after freezing/unfreezing parameters in get_learned_equations
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
        self.optimizer = AdamW(trainable_params, lr=fine_tune_lr)
        
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
            'monitor': 'symbolic/val_loss',  # Monitor fine-tuning val loss
            'interval': 'epoch',
            'frequency': 1
        }
        
        # Update optimizer and scheduler in model
        self.model.optimizer = self.optimizer
        self.model.scheduler = self.scheduler
        
        # Rebuild trainer with new configuration for fine-tuning
        early_stopping = EarlyStopping(
            monitor='symbolic/val_loss',  # Monitor fine-tuning val loss
            patience=self.cfg.patience, 
            verbose=True,
            mode='min'
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            monitor='symbolic/val_loss',  # Monitor fine-tuning val loss
            filename='best_model', 
            save_top_k=1, 
            mode='min', 
            verbose=True,
            save_last=False,
            enable_version_counter=False  # Prevent version suffixes
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
            # gradient_clip_val=0.5
        )
        
        # Fine-tune
        self.trainer.fit(self.model, train_dataloader, val_dataloader)
        
        print("Fine-tuning completed!")
        print(f"Best model updated at: {self.checkpoint_dir}/best_model.ckpt")
        
        # Get symbolic equations after fine-tuning
        self.model.model.get_learned_equations(log_dir, fine_tuned=True)      

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