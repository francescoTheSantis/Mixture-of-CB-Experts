import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch
from torch.optim import AdamW
import numpy as np
import pandas as pd
from src.metrics import f1_acc_metrics
from tqdm import tqdm
from src.utils.scalers import StandardScaler
from src.utilities import symbolic_regression

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
        self.scale_variables = cfg.scale_variables if 'scale_variables' in cfg else True

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

        # If regression, standardize the target variable and concepts, and store the scalers in the model
        if self.model.model.task == 'regression' and self.scale_variables:
            # Fit scaler to y data
            y_scaler = StandardScaler(dims=(0,))
            y_scaler.fit(y_trues)

            # Store the y scaler in the engine & model
            self.model.y_scaler = y_scaler
            self.model.model.y_scaler = y_scaler
            
            # Fit scalers to concept data (one per concept)
            n_concepts = c_trues.shape[1]
            c_scalers = []
            for i in range(n_concepts):
                c_scaler = StandardScaler(dims=(0,))
                c_scaler.fit(c_trues[:, i:i+1])
                c_scalers.append(c_scaler)
            
            # Store the concept scalers in the engine & model
            self.model.c_scalers = c_scalers
            self.model.model.c_scalers = c_scalers
        else:
            self.model.y_scaler = None
            self.model.model.y_scaler = None
            self.model.c_scalers = None
            self.model.model.c_scalers = None

        if self.model.model.__class__.__name__ == 'KANSymbolicCBM':
            # Scale concepts if needed before setting up KAN grid
            kan_inputs_to_use = c_trues.clone().to(self.cfg.gpus[0])
            if self.model.model.task == 'regression' and self.scale_variables:
                for i, c_scaler in enumerate(self.model.c_scalers):
                    kan_inputs_to_use[:, i:i+1] = c_scaler.transform(kan_inputs_to_use[:, i:i+1])
            
            self.kan_inputs = kan_inputs_to_use
            self.model.model.setup_kan_grid(self.kan_inputs)
            # Save scaled c_trues in model as it will be used to update the grid during training
            self.model.grid_inputs = kan_inputs_to_use

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

        # Set allow symbolic flag in the model
        self.model.model.allow_symbolic = True

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
            # NOTE: if you want, you can prune the KAN layers before allowing symbolic execution.
            # Unfortunatelly, the pruning does not work when speed_up_training=True.
            # So, if you want to prune, set speed_up_training=False in the model config.
            if self.model.model.speed_up_training:
                self.model.model.allow_symbolic()
                print("Skipping pruning as speed_up_training=True")
                epochs = 1
            else:
                print("Pruning KAN layers before allowing symbolic execution")
                self.model.model.prune()
                epochs = self.cfg.max_epochs

            # Update the grid (self.kan_inputs is already scaled from train function)
            self.model.model.setup_kan_grid(self.kan_inputs) 

        # For SR-Sym-CBM, collect data for symbolic fine-tuning
        elif model_name == 'sr_symbolic_cbm':
            epochs = self.cfg.max_epochs
            self.model.eval()
            self.model = self.model.to(self.cfg.gpus[0])
            stored_concepts = []
            stored_targets = []
            stored_selector_probs = []
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
                    output = self.model.model.forward(inputs, store_for_finetuning=True)

                    # Add to lists
                    # if the problem is regression, learn the true targets, if classification, learn the logits (y_hat)
                    if self.cfg.dataset.metadata.task == 'regression':
                        stored_targets.append(y.detach().cpu())
                    else:
                        # stored_targets.append(output['y_hat'].detach().cpu())
                        
                        # For classification, store the true labels as targets
                        # The true labels need to be reshaped to (bsz, n_classes) one-hot encoding
                        n_classes = output['y_hat'].shape[1]
                        y_one_hot = torch.zeros(y.size(0), n_classes)
                        y_one_hot.scatter_(1, y.to('cpu').view(-1, 1).long(), 1)

                        # From [0,1] to [-1,1]
                        y_one_hot = y_one_hot * 2 - 1

                        stored_targets.append(y_one_hot.detach().cpu())

                    # If the training is disjoint use the true concepts for SR algorithm, otherwise use the predicted concepts.
                    concepts_for_sr_algorithm = c if self.cfg.disjoint_training else output['c_hat']
                    stored_concepts.append(concepts_for_sr_algorithm.detach().cpu())
                    stored_selector_probs.append(output['sampled_memory_idxs'].detach().cpu())

                    # Clear GPU memory
                    del x, c, y, inputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Concatenate stored data
            concatenated_concepts = torch.cat(stored_concepts, dim=0)
            concatenated_targets = torch.cat(stored_targets, dim=0)
            concatenated_selector_probs = torch.cat(stored_selector_probs, dim=0)
            
            # Scale concepts and targets if scale_variables is True and task is regression
            if self.cfg.dataset.metadata.task == 'regression' and self.scale_variables:
                # Scale targets
                concatenated_targets = self.model.y_scaler.transform(concatenated_targets)
                
                # Scale concepts (one by one using per-concept scalers)
                for i, c_scaler in enumerate(self.model.c_scalers):
                    concatenated_concepts[:, i:i+1] = c_scaler.transform(concatenated_concepts[:, i:i+1])
            
            # Extract equations using symbolic regression
            print("Extracting symbolic equations from stored data...")
            equations = symbolic_regression(
                stored_concepts=concatenated_concepts,
                stored_targets=concatenated_targets,
                stored_selector_probs=concatenated_selector_probs,
                memory_size=self.model.model.memory_size,
                output_size=self.model.model.output_size,
                c_names=self.model.model.c_names,
                y_names=self.model.model.y_names,
                device=self.cfg.gpus[0],
                pysr_params=self.model.model.pysr_params,
                task=self.cfg.dataset.metadata.task,
                disjoint_training=self.cfg.disjoint_training
            )

            # Run symbolic substitution to create the SymbolicPredictor
            self.model.model.symbolic_substitution(equations)

        elif model_name in ['memory_cbm', 'linear_symbolic_cbm']:
            print("Cutting parameters of the predictor below the threshold: ", self.model.model.threshold)
            self.model.model.cut_weights()
            epochs = self.cfg.max_epochs
        
        # Set fine-tuning mode to change metric names
        self.model.fine_tuning = True
        self.model.fine_tuning_stage = 'allow_symbolic'
        self.model._set_metrics()
        
        if model_name == 'sr_symbolic_cbm':
            # Higher LR for SR-Sym-CBM as there only few parameters in the predictor
            fine_tune_lr = self.cfg.dataset.metadata.lr * 5 
        else:
            fine_tune_lr = self.cfg.dataset.metadata.lr
        
        # Recreate optimizer to include new SymbolicPredictor parameters
        # The predictor was just replaced in symbolic_substitution, so we need to
        # rebuild the optimizer to include its trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Number of trainable parameters after symbolic substitution: {sum(p.numel() for p in trainable_params)}")
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

                    if self.cfg.dataset.metadata.task == 'regression' and self.scale_variables:
                        # If regression, inverse transform the predictions
                        y_preds = self.model.y_scaler.inverse_transform(y_preds)

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