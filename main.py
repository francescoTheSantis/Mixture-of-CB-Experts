import hydra
from omegaconf import DictConfig, open_dict
from hydra.utils import instantiate
import torch
import os

from src.trainer import Trainer
from src.utilities import set_seed, set_loggers, set_matmul_precision
from src.utilities import update_config_from_data, is_valid_experiment, \
    generate_data_path, save_licem_linear_coefficients

@hydra.main(config_path="conf", config_name="sr_ablation")
def main(cfg: DictConfig) -> None:

    # Set the matmul precision according to the device
    set_matmul_precision()

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
        loaded_train = torch.load(train_path)
        loaded_val = torch.load(val_path)
        loaded_test = torch.load(test_path)

    # Otherwise, preprocess the data and then store the results
    else:
        print('Prepearing dataloaders...')
        os.makedirs(data_path, exist_ok=True)
        loaded_train, loaded_val, loaded_test = loader.load_data(cfg)
        torch.save(loaded_train, train_path)
        torch.save(loaded_val, val_path)
        torch.save(loaded_test, test_path)

    # If the config is meant to just generate and store the dataset, exit here
    if cfg.only_store_dataset:
        print('Dataset stored. Exiting...')
        return

    # Load the concept names and groups
    c_names, y_names, c_groups = loader.get_names(cfg)

    # Set the c_names and y_names in the config
    cfg = update_config_from_data(cfg, loaded_train, c_names, y_names, c_groups, csv_logger.log_dir)

    # Check whether it is a valid combination of dataset and model.
    # Some models (e.g., dcr) cannot be executed on some datasets (e.g., mnist_arithmetic).
    is_valid_experiment(cfg)    

    ###### Instantiate the model ######
    model = instantiate(cfg.engine)

    ###### Training ######
    # Initialize the trainer
    trainer = Trainer(model, cfg, wandb_logger, csv_logger)
    trainer.build_trainer()

    # Train the model
    trainer.train(loaded_train, loaded_val)

    ###### Fine-tuning for symbolic cmr (kan implementation) model ######
    if cfg.model.metadata.name in ['kan_symbolic_cbm', 'sr_symbolic_cbm']:
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

    ###### Store symbolic expression associated to the predictor ######
    file_name = f"{log_dir}/equations"
    os.makedirs(file_name, exist_ok=True)
    model.model.get_symbolic_equivalent(file_name)

    ###### Perform Interventions ######
    if model.model.has_concepts:
        intervention_df = trainer.interventions(loaded_test)
        intervention_df.to_csv(f"{log_dir}/interventions.csv", index=False)

    ###### Save the learned linear coefficients (Useful for showing explanations) ######
    if cfg.model.metadata.name == 'licem':
        save_licem_linear_coefficients(model, loaded_train, log_dir, split='train')
        save_licem_linear_coefficients(model, loaded_test, log_dir, split='test')
        # save the c_names and y_names
        with open(f"{log_dir}/c_names.txt", "w") as f:
            for c in c_names:
                f.write(f"{c}\n")
        with open(f"{log_dir}/y_names.txt", "w") as f:
            for y in y_names:
                f.write(f"{y}\n")

    # Close the wandb logger if it is used
    if wandb_logger is not None:
        wandb_logger.experiment.finish()

if __name__ == "__main__":
    set_matmul_precision()
    main()