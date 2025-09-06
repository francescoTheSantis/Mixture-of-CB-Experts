from src.trainer import Trainer
import hydra
from omegaconf import DictConfig, open_dict
from hydra.utils import instantiate
from src.utilities import set_seed, set_loggers
import torch
import os
from env import CACHE
from src.utilities import update_config_from_data, is_valid_experiment, generate_data_path

@hydra.main(config_path="conf", config_name="test")
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
            data_path = data_path
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

    # Load the concept names and groups
    c_names, y_names, c_groups = loader.get_names()

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

    ###### Test ######
    # Test the model on the test-set
    trainer.test(loaded_test)

    if model.model.has_concepts:
        ###### Perform Intervetions ######
        intervention_df = trainer.interventions(loaded_test)
        intervention_df.to_csv(f"{log_dir}/interventions.csv", index=False)

    # Close the wandb logger if it is used
    if wandb_logger is not None:
        wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()