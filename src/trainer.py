import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch
from torch.optim import AdamW
import numpy as np
import pandas as pd
from src.metrics import f1_acc_metrics
from tqdm import tqdm
from src.models.m_licem import LinearMemoryReasoner

class Trainer:
    """
    Trainer class for the pytorch_lightning model.
    """
    def __init__(self, model, cfg, wandb_logger, csv_logger):
        self.cfg = cfg
        self.wandb_logger = wandb_logger
        self.csv_logger = csv_logger
        self.model = model
        self.model = self.model.to(self.cfg.gpus[0]) # Move the model to the GPU
        #self.epss = np.arange(0, 1.1, 0.25) # Noise levels for interventions
        self.epss = [0]
        self.p_ints = np.arange(0, 1.1, 0.1) # Intervention probabilities

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
            gradient_clip_val=1.0
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
        self.trainer.fit(self.model, 
                         train_dataloader, 
                         val_dataloader, ckpt_path=ckpt_path)

    def test(self, test_dataloader, ckpt_path=None):
        # Load the best model and test
        if ckpt_path is None:
            ckpt_path = self.trainer.checkpoint_callback.best_model_path
        self.trainer.test(self.model, test_dataloader, ckpt_path=ckpt_path)

    def interventions(self, test_dataloader, verbose=True):
        """
        Perform interventions on the test set and return the dataframe containing the results.
        Interventional accuracy is computed for different levels of noise and intervention probability.
        """
        intervention_df = pd.DataFrame(columns=['noise', 'p_int', 'f1', 'accuracy'])
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
                    for batch in test_dataloader:
                        x, c, y = self.model.unpack_batch(batch)
                        # Move the data to the GPU
                        if isinstance(x, dict):
                            x = {k: v.to(self.cfg.gpus[0]) for k, v in x.items()}
                        else:  
                            x = x.to(self.cfg.gpus[0])
                        c = c.to(self.cfg.gpus[0])
                        y = y.to(self.cfg.gpus[0])
                        inputs = {'x':x, 'c':c, 'y':y}
                        self.model.model.int_prob = p_int
                        output = self.model.forward(inputs)
                        output = self.model.model.filter_output_for_metrics(*output)
                        y_pred = output[0]
                        y_preds.append(y_pred)
                        y_trues.append(y)
                    y = torch.cat(y_trues, dim=0)
                    y_preds = torch.cat(y_preds, dim=0)
                    y = y.cpu().numpy()
                    if len(self.cfg.model.params.y_names)==1 and self.cfg.dataset.metadata.task != 'regression':
                        if self.model.model.__class__.__name__ in ['DeepConceptReasoner', 'ConceptMemoryReasoner']:
                            y_preds = (y_preds > 0.5).long().cpu().numpy()
                        else:
                            y_preds = (y_preds > 0.).long().cpu().numpy()
                    if self.cfg.dataset.metadata.task == 'regression':
                        y_preds = y_preds.squeeze().cpu().numpy()
                    else:
                        y_preds = y_preds.argmax(-1).cpu().numpy()

                    if self.cfg.dataset.metadata.task == 'regression':
                        task_f1, task_acc = 0, 0
                        mse = np.mean((y - y_preds) ** 2)
                    else:
                        task_f1, task_acc = f1_acc_metrics(y, y_preds)
                        mse = 0
                    intervention_results = {
                        'noise': round(eps,1), 
                        'p_int': round(p_int,1), 
                        'f1': round(task_f1,2), 
                        'accuracy': round(task_acc,2),
                        'mse': round(mse, 4)
                    }
                    intervention_df = pd.concat([intervention_df, pd.DataFrame([intervention_results])], ignore_index=True)
        self.model.model.test_interventions = False
        return intervention_df

    def plot_results(self, test_dataloader, verbose=True):
        """
        Plot the results of the model on the test set.
        This method is specific to the xor, or, nor, and, nand, xnor datasets.
        If the model is a LinearModel it will plot the weights of the model as well as the predictions.
        """

        # Set the model on the right device
        from matplotlib import pyplot as plt
        import numpy as np

        self.model = self.model.to(self.cfg.gpus[0])
        self.model.eval()

        with torch.no_grad():
            input_list = []
            outputs = []
            for batch in test_dataloader:
                x, c, y = self.model.unpack_batch(batch)
                # Move the data to the GPU
                x = x.to(self.cfg.gpus[0])
                c = c.to(self.cfg.gpus[0])
                y = y.to(self.cfg.gpus[0])
                inputs = {'x': x, 'c': c, 'y': y}
                output = self.model.forward(inputs)
                output = self.model.model.filter_output_for_metrics(*output)
                y_pred = output[0]
                y_pred = y_pred.cpu().numpy()
                if len(self.cfg.model.params.y_names) == 1:
                    y_pred = (y_pred > 0.5).astype(int)
                else:
                    y_pred = y_pred.argmax(-1)
                outputs.append(y_pred)
                input_list.append(x.cpu().numpy())

            # Plot the results
            outputs = np.concatenate(outputs, axis=0).squeeze()
            input_list = np.concatenate(input_list, axis=0)

            pos_preds = input_list[outputs == 1]
            neg_preds = input_list[outputs == 0]

            if isinstance(self.model.model, LinearMemoryReasoner):
                # Plot the weights of the model
                model = self.model.model
                weights = model.equation_decoder(model.equation_memory.weight).detach().cpu().numpy()
                norm_weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
                for i, (w1, w2) in enumerate(zip(norm_weights[:, 0], norm_weights[:, 1])):
                    plt.arrow(0, 0, w1.item(), w2.item(), head_width=0.05, head_length=0.1, fc='k', ec='k', label=f'Weight {i+1}')

                if model.negative_concepts:
                    pos_preds = 2* pos_preds - 1
                    neg_preds = 2* neg_preds - 1

            plt.scatter(neg_preds[:, 0], neg_preds[:, 1], label='Negative Predictions')
            plt.scatter(pos_preds[:, 0], pos_preds[:, 1], label='Positive Predictions')

            plt.xlabel('X1')
            plt.ylabel('X2')

            plt.title('Model Predictions on Test Set')
            plt.legend()
            plt.savefig(f"{self.wandb_logger.experiment.dir}/test_predictions.png")

            if verbose:
                plt.show()
            else:
                plt.close()
