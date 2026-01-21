import hydra
from omegaconf import DictConfig, open_dict
from hydra.utils import instantiate
import torch
import os
from tqdm.auto import tqdm
import copy
from pathlib import Path

from src.trainer import Trainer
from src.utilities import set_seed, set_loggers, update_config_from_data, \
    is_valid_experiment, generate_data_path


@hydra.main(config_path="conf", config_name="adaptability_experiment")
def main(cfg: DictConfig) -> None:
    """
    Multi-constraint symbolic regression experiment.
    
    This script performs:
    1. Initial training once
    2. For each constraint set in the configuration:
       - Apply symbolic regression with those constraints
       - Fine-tune the model
       - Test on the test set
       - Save results separately for each constraint
    """
    
    # Verify this is a symbolic model
    if cfg.model.metadata.name != 'sr_symbolic_cbm':
        raise ValueError(
            f"Adaptability experiment only supports symbolic models. "
            f"Got: {cfg.model.metadata.name}"
        )
    
    # Verify multi-constraint flag is enabled
    if not cfg.get('multi_constraint_experiment', False):
        raise ValueError(
            "multi_constraint_experiment flag must be set to True in config. "
            "Use a regular main.py for standard experiments."
        )
    
    # Get constraint sets from configuration
    constraint_sets = cfg.get('constraint_sets', None)
    if constraint_sets is None or len(constraint_sets) == 0:
        raise ValueError(
            "No constraint_sets defined in configuration. "
            "Add a 'constraint_sets' list with different pysr_params configurations."
        )
    
    print("="*70)
    print("MULTI-CONSTRAINT SYMBOLIC REGRESSION EXPERIMENT")
    print("="*70)
    print(f"Model: {cfg.model.metadata.name}")
    print(f"Number of constraint sets: {len(constraint_sets)}")
    print("="*70)
    
    # Initialize the wandb logger
    wandb_logger, csv_logger = set_loggers(cfg)
    base_log_dir = csv_logger.log_dir
    
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
    
    # Load or preprocess data
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
    else:
        print('Preparing dataloaders...')
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
    cfg = update_config_from_data(cfg, loaded_train, c_names, y_names, c_groups, base_log_dir)
    
    # Check whether it is a valid combination of dataset and model
    is_valid_experiment(cfg)
    
    ###### Instantiate the model ######
    model = instantiate(cfg.engine)
    
    ###### PHASE 0: Initial Training (performed once) ######
    print("\n" + "="*70)
    print("PHASE 0: INITIAL TRAINING (performed once)")
    print("="*70)
    
    # Initialize the trainer
    trainer = Trainer(model, cfg, wandb_logger, csv_logger)
    trainer.build_trainer()
    
    # Train the model
    trainer.train(loaded_train, loaded_val)
    
    # Save the initial trained checkpoint with a protected name
    # Copy best_model.ckpt to initial_model.ckpt to prevent overwriting
    original_checkpoint = f"{base_log_dir}/best_model.ckpt"
    initial_checkpoint_path = f"{base_log_dir}/initial_model.ckpt"
    
    if os.path.exists(original_checkpoint):
        import shutil
        shutil.copy2(original_checkpoint, initial_checkpoint_path)
        print(f"\nInitial training completed. Checkpoint saved at: {initial_checkpoint_path}")
    else:
        raise FileNotFoundError(f"Expected checkpoint not found at: {original_checkpoint}")
    
        
    # Test the model with current symbolic equations
    test_checkpoint = f"{constraint_log_dir}/best_model.ckpt"
    trainer.test(loaded_test, ckpt_path=test_checkpoint)
    
    ###### Save Constraint-Specific Results ######
    print(f"\nSaving results for blackbox")
    
    # Perform interventions if applicable
    if model.model.has_concepts:
        intervention_df = trainer.interventions(loaded_test)
        intervention_df.to_csv(
            f"{constraint_log_dir}/interventions.csv", 
            index=False
        )

    ###### LOOP OVER CONSTRAINT SETS ######
    all_results = []
    
    for constraint_idx, constraint_config in enumerate(constraint_sets):
        print("\n" + "="*70)
        print(f"CONSTRAINT SET {constraint_idx + 1}/{len(constraint_sets)}")
        print("="*70)
        print(f"Constraint configuration: {constraint_config}")
        print("="*70)
        
        # Create subdirectory for this constraint set
        # Include constraint name in directory if available
        constraint_name = constraint_config.get('name', f'{constraint_idx}')
        constraint_log_dir = os.path.join(base_log_dir, f"constraint_{constraint_idx}_{constraint_name}")
        os.makedirs(constraint_log_dir, exist_ok=True)
        
        # Reload the model from the initial checkpoint for each constraint
        print(f"\nReloading model from initial checkpoint: {initial_checkpoint_path}")
        checkpoint = torch.load(initial_checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        
        # Update the model's pysr_params with the constraint configuration
        if hasattr(model.model, 'pysr_params'):
            original_pysr_params = copy.deepcopy(model.model.pysr_params)
            
            # Apply constraint overrides
            print(f"Applying constraints to pysr_params...")
            for key, value in constraint_config.items():
                model.model.pysr_params[key] = value
                print(f"  {key}: {value}")
            
            print(f"\nUpdated pysr_params: {model.model.pysr_params}")
        else:
            print("Warning: Model does not have pysr_params attribute")
        
        # Update checkpoint directory for this constraint
        trainer.checkpoint_dir = constraint_log_dir
        
        ###### PHASE 1: Symbolic Regression with Current Constraints ######
        print("\n" + "-"*70)
        print(f"PHASE 1: SYMBOLIC REGRESSION (Constraint Set {constraint_idx + 1})")
        print("-"*70)
        
        # Run symbolic discovery with current constraints
        trainer.allow_symbolic(loaded_train, loaded_val)
        
        # For KAN models, perform additional fine-tuning
        if cfg.model.metadata.name == 'kan_symbolic_cbm':
            print("\n" + "-"*70)
            print(f"PHASE 2: FINE-TUNING (Constraint Set {constraint_idx + 1})")
            print("-"*70)
            trainer.fine_tune(
                loaded_train, 
                loaded_val,
                log_dir=constraint_log_dir,
            )
        
        ###### PHASE 2: Testing with Current Constraints ######
        print("\n" + "-"*70)
        print(f"PHASE 3: TESTING (Constraint Set {constraint_idx + 1})")
        print("-"*70)
        
        # Test the model with current symbolic equations
        test_checkpoint = f"{constraint_log_dir}/best_model.ckpt"
        trainer.test(loaded_test, ckpt_path=test_checkpoint)
        
        ###### Save Constraint-Specific Results ######
        print(f"\nSaving results for constraint set {constraint_idx + 1}...")
        
        # Perform interventions if applicable
        if model.model.has_concepts:
            intervention_df = trainer.interventions(loaded_test)
            intervention_df.to_csv(
                f"{constraint_log_dir}/interventions.csv", 
                index=False
            )
        
        # Save constraint configuration
        import json
        with open(f"{constraint_log_dir}/constraint_config.json", 'w') as f:
            # Convert to regular dict for JSON serialization
            constraint_dict = dict(constraint_config)
            json.dump(constraint_dict, f, indent=2)
        
        # Store summary results
        all_results.append({
            'constraint_idx': constraint_idx,
            'constraint_config': dict(constraint_config),
            'log_dir': constraint_log_dir,
            'checkpoint': test_checkpoint,
        })
        
        # Restore original pysr_params for next iteration
        if hasattr(model.model, 'pysr_params'):
            model.model.pysr_params = original_pysr_params
        
        print(f"\n✓ Constraint set {constraint_idx + 1} completed!")
        print(f"  Results saved in: {constraint_log_dir}")
    
    ###### Save Overall Summary ######
    print("\n" + "="*70)
    print("ALL CONSTRAINT SETS COMPLETED")
    print("="*70)
    
    # Save summary of all experiments
    import pandas as pd
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(base_log_dir, "constraint_sets_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    # Print all result directories
    print("\nResults for each constraint set:")
    for i, result in enumerate(all_results):
        print(f"  [{i+1}] {result['log_dir']}")
    
    # Close the wandb logger if it is used
    if wandb_logger is not None:
        wandb_logger.experiment.finish()
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*70)


if __name__ == "__main__":
    main()
