from src.trainer import Trainer
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate, get_class
from src.utilities import set_seed, set_loggers
import torch
import os
from env import CACHE
from src.utilities import update_config_from_data

@hydra.main(config_path="conf", config_name="test")
def main(cfg: DictConfig) -> None:

    # Initialize the wandb logger
    wandb_logger, csv_logger = set_loggers(cfg)

    print("Configuration Parameters:")
    for key, value in cfg.items():
        print(f"{key}: {value}")
    print('\n')

    # Set the seed
    set_seed(cfg.seed)

    ###### Load the data ######
    data_path = os.path.join(str(CACHE), 'stored_tensors', cfg.dataset.metadata.name)
    train_path = f"{data_path}/train.pt"
    val_path = f"{data_path}/val.pt"
    test_path = f"{data_path}/test.pt"
    # If the data have been preprocessed, load the preprocessed data
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print('Loading pre-processed data...')
        loaded_train = torch.load(f"{data_path}/train.pt")
        loaded_val = torch.load(f"{data_path}/val.pt")
        loaded_test = torch.load(f"{data_path}/test.pt")
        with open(os.path.join(data_path, "lists.txt"), "r") as file:
            lines = file.readlines()
            c_names = lines[1].strip().split(", ")
            y_names = lines[3].strip().split(", ")
    # Otherwise, preprocess the data and then store them
    else:
        print('Preprocessing data...')
        loader = instantiate(cfg.dataset.loader)
        loaded_train, loaded_val, loaded_test, c_names, y_names = loader.load_data()
        os.makedirs(data_path, exist_ok=True)
        torch.save(loaded_train, train_path)
        torch.save(loaded_val, val_path)
        torch.save(loaded_test, test_path)

        with open(os.path.join(data_path, "lists.txt"), "w") as file:
            file.write("c_names 1:\n")
            file.write(", ".join(map(str, c_names)) + "\n")
            file.write("y_names:\n")
            file.write(", ".join(y_names) + "\n") 

    # Set the c_names and y_names in the config
    cfg = update_config_from_data(cfg, loaded_train, c_names, y_names)

    ###### Instantiate the model ######
    model = instantiate(cfg.engine)

    ###### Training ######
    # Initialize the trainer
    trainer = Trainer(model, cfg, wandb_logger, csv_logger)
    trainer.build_trainer()

    # Train the model
    trainer.train(loaded_train, loaded_val)

    # Load the best model
    model_class = get_class(cfg.model.params._target_)
    model_kwargs = {k: v for k, v in cfg.model.params.items() if k not in ["_target_"]}
    model = model_class.load_from_checkpoint(trainer.trainer.checkpoint_callback.best_model_path, **model_kwargs)
    trainer.model = model

    ###### Test ######
    # Test the model on the test-set
    trainer.test(loaded_test)

    if model.has_concepts:
        ###### Perform Intervetions ######
        intervention_df = trainer.interventions(loaded_test)
        log_dir = csv_logger.log_dir
        intervention_df.to_csv(f"{log_dir}/interventions.csv", index=False)

        ###### Compute Concept Alignment Score (CAS) ######


        ###### Compute CACE ######


    # Close the wandb logger if it is used
    if wandb_logger is not None:
        wandb_logger.experiment.finish()

if __name__ == "__main__":
    main()
