import hydra
from omegaconf import DictConfig, open_dict
from hydra.utils import instantiate
import torch
import os
import numpy as np
from tqdm.auto import tqdm

from src.trainer import Trainer
from src.utilities import set_seed, set_loggers, update_config_from_data, \
    is_valid_experiment, generate_data_path

@hydra.main(config_path="conf", config_name="debugging")
def main(cfg: DictConfig) -> None:

    # Initialize the wandb logger
    wandb_logger, csv_logger = set_loggers(cfg)

    # Get the log_dir
    log_dir = csv_logger.log_dir

    # Set the seed
    set_seed(cfg.seed)

    ###### Load the data ######
    data_path, train_path, val_path, test_path = generate_data_path(cfg)

    # Add data_path to the loader and engine configs
    with open_dict(cfg):
        cfg.dataset.loader.update(
            data_path = data_path,
            use_stored_dataset = cfg.use_stored_dataset
        )
        cfg.engine.update(
            data_path = data_path
        )

    # Loader instantiation
    loader = instantiate(cfg.dataset.loader)

    # If the data have been preprocessed and use_stored_dataset=True, load the preprocessed data
    if os.path.exists(train_path) and os.path.exists(val_path)\
                                  and os.path.exists(test_path)\
                                  and cfg.use_stored_dataset:
        print('Loading pre-processed data...')
        with tqdm(total=3, desc="Loading datasets") as pbar:
            loaded_train = torch.load(train_path)
            pbar.update(1)
            loaded_val = torch.load(val_path)
            pbar.update(1)
            loaded_test = torch.load(test_path)
            pbar.update(1)

    # Otherwise, preprocess the data and then store the results
    else:
        print('Prepearing dataloaders...')
        os.makedirs(data_path, exist_ok=True)
        loaded_train, loaded_val, loaded_test = loader.load_data(cfg)

        print('Saving preprocessed data...')
        with tqdm(total=3, desc="Saving datasets") as pbar:
            torch.save(loaded_train, train_path)
            pbar.update(1)
            torch.save(loaded_val, val_path)
            pbar.update(1) 
            torch.save(loaded_test, test_path)
            pbar.update(1)

    # If the config is meant to just generate and store the dataset, exit here
    if cfg.only_store_dataset:
        print('Dataset stored. Exiting...')
        return

    # Load the concept names and groups
    c_names, y_names, c_groups = loader.get_names(cfg)

    # Set the c_names and y_names in the config
    cfg = update_config_from_data(cfg, loaded_train, c_names, y_names, c_groups, csv_logger.log_dir)

    # Check whether it is a valid combination of dataset and model.
    # Some models (e.g., dcr) cannot be executed on regression datasets.
    is_valid_experiment(cfg)    

    ###### Instantiate the model ######
    model = instantiate(cfg.engine)

    ###### Training ######
    # Initialize the trainer
    trainer = Trainer(model, cfg, wandb_logger, csv_logger)
    trainer.build_trainer()

    # Train the model
    trainer.train(loaded_train, loaded_val)

    ###### Fine-tuning for symbolic models ######
    if cfg.model.metadata.name in ['kan_symbolic_cbm', 'sr_symbolic_cbm', 'memory_cbm', 'linear_symbolic_cbm', 'bool_symbolic_cbm']:
        # Phase 1: Allow symbolic execution
        print("\n" + "="*70)
        print("PHASE 1: Allow symbolic execution")
        print("="*70)
        trainer.allow_symbolic(
            loaded_train, 
            loaded_val
        )

    if cfg.model.metadata.name == 'kan_symbolic_cbm':
        # Phase 2: Fine-tuning with symbolic expressions
        # This phase is needed just for the model using KAN layers as task predictors
        print("\n" + "="*70)
        print("PHASE 2: Fine-tuning with symbolic expressions")
        print("="*70)
        trainer.fine_tune(
            loaded_train, 
            loaded_val,
            log_dir=log_dir,  # where equations are stored
        )

    ###### Testing ######
    # Test the model on the test-set
    trainer.test(loaded_test)

    ###### Perform Interventions ######
    if model.model.has_concepts:
        intervention_df = trainer.interventions(loaded_test)
        intervention_df.to_csv(f"{log_dir}/interventions.csv", index=False)

    ###### Global Interpretability Evaluation ######
    if cfg.model.metadata.name in ['licem', 'linear_symbolic_cbm']:
        gi_df = trainer.global_interpretability(loaded_train, loaded_test)
        gi_df.to_csv(f"{log_dir}/global_interpretability.csv", index=False)

    ###### Explanation Robustness ######
    if cfg.model.metadata.name in ['licem', 'linear_symbolic_cbm']:
        robustness_alphas = list(cfg.robustness_alphas) if 'robustness_alphas' in cfg \
            else list(np.arange(0.01, 0.16, 0.01))
        robustness_K = cfg.robustness_K if 'robustness_K' in cfg else 10
        robustness_df = trainer.explanation_robustness(
            loaded_test, alphas=robustness_alphas, K=robustness_K
        )
        robustness_df.to_csv(f"{log_dir}/explanation_robustness.csv", index=False)

    # Close the wandb logger if it is used
    if wandb_logger is not None:
        wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()