from src.trainer import Trainer
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.utilities import set_seed, set_loggers
import torch
import os
from env import CACHE
from src.utilities import update_config_from_data

@hydra.main(config_path="conf", config_name="sweep")
def main(cfg: DictConfig) -> None:

    # Initialize the wandb logger
    wandb_logger, csv_logger = set_loggers(cfg)

    # Set the seed
    set_seed(cfg.seed)

    ###### Load the data ######
    data_path = os.path.join(str(CACHE), 'stored_tensors', cfg.dataset.metadata.name)
    train_path = f"{data_path}/train.pt"
    val_path = f"{data_path}/val.pt"
    test_path = f"{data_path}/test.pt"

    # Loader instantiation
    loader = instantiate(cfg.dataset.loader)

    # If the data have been preprocessed and use_stored_dataset=True, load the preprocessed data
    if os.path.exists(train_path) and os.path.exists(val_path)\
                                  and os.path.exists(test_path)\
                                  and cfg.use_stored_dataset\
                                  and loader.extract_embeddings:
        print('Loading pre-processed data...')
        loaded_train = torch.load(f"{data_path}/train.pt")
        loaded_val = torch.load(f"{data_path}/val.pt")
        loaded_test = torch.load(f"{data_path}/test.pt")
    # Otherwise, preprocess the data and then store the results
    else:
        print('Preprocessing data...')
        loaded_train, loaded_val, loaded_test = loader.load_data()
        if loader.extract_embeddings:
            os.makedirs(data_path, exist_ok=True)
            torch.save(loaded_train, train_path)
            torch.save(loaded_val, val_path)
            torch.save(loaded_test, test_path)

    # Load the concept names and groups
    c_names, y_names, c_groups = loader.get_names()

    # Set the c_names and y_names in the config
    cfg = update_config_from_data(cfg, loaded_train, c_names, y_names, c_groups, csv_logger.log_dir)

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
        log_dir = csv_logger.log_dir
        intervention_df.to_csv(f"{log_dir}/interventions.csv", index=False)

    # Close the wandb logger if it is used
    if wandb_logger is not None:
        wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()