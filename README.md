# Linear-Memory-Reasoner

A framework for training and evaluating concept-based reasoning models with linear memory complexity.

## Setup

1. **Create and activate the Conda environment**:
    ```bash
    conda env create -f environment.yml
    conda activate lmr
    ```

2. **Configure environment variables in** `env.py`:
    - `HOME` - Path to the project root
    - `DATA_PATH` - Path to load datasets
    - `PROJECT_NAME` (optional) - Project identifier for logging

3. **Datasets**:
    Most datasets are automatically downloaded and processed when needed. For CIFAR10/CIFAR100, you need to manually download prerequisite text files (one-time setup):
    - Download `cifar10_filtered.txt` and `cifar10_classes.txt` from [Label-free-CBM](https://github.com/Trustworthy-ML-Lab/Label-free-CBM/tree/main)
    - Place them in `{DATA_PATH}/cifar10/` directory
    - Same for CIFAR100: download and place in `{DATA_PATH}/cifar100/`

## Running Experiments

To replicate all experiments from the paper:

1. **Run multiple experiments**:
    ```bash
    python main.py --config-name "specific_config"
    ```
    Where "specific_config" is one of the configurations present in the `conf/` folder.

2. **Plot results**:
    Add to the "paths" list in the `show_results.py` script, the paths containing the results obtained by running the previous configurations.Then run the following code: 
    ```bash
    python show_results.py
    ```
    Edit `custom_order` and `model_styles` in `show_results.py` to customize plots.

## Configuration

Modify training settings in `conf/`:
- `dataset/` - Dataset parameters and preprocessing
- `model/` - Model architectures and hyperparameters
- `encoder/` - Inpur encoder configurations
- `engine/` - Training loop settings
- `common.yaml` - Shared defaults

## Results

Experiment logs and checkpoints are saved in the output directory specified by Hydra (default: `outputs/`).