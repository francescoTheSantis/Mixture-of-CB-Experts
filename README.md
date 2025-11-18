Here is an updated version of your README with a **very short explanation** of why the environment must be created in two steps.
I kept it minimal and non-technical, as you requested, but accurate:

---

# Linear-Memory-Reasoner

A framework for training and evaluating concept-based reasoning models with linear memory complexity.

## Setup

1. **Create and activate the Conda environment**
   *(Environment creation is done in two steps because one dependency (`pytorch_concepts`) imports PyTorch during installation, and PyTorch is only available after the base Conda environment is created.)*

   ```bash
   conda env create -f environment.yml
   conda activate lmr
   ```

2. **Install the remaining dependency**

   ```bash
   pip install --no-build-isolation git+https://github.com/pyc-team/pytorch_concepts.git@models
   ```

3. **Configure environment variables in** `env.py`:

   * `HOME` - Path to the project root
   * `DATA_PATH` - Path to load datasets
   * `PROJECT_NAME` (optional) - Project identifier for logging

4. **Datasets**:
   Most datasets are automatically downloaded and processed when needed. For CIFAR10/CIFAR100, you need to manually download prerequisite text files (one-time setup):

   * Download `cifar10_filtered.txt` and `cifar10_classes.txt` from [Label-free-CBM](https://github.com/Trustworthy-ML-Lab/Label-free-CBM/tree/main)
   * Place them in `{DATA_PATH}/cifar10/`
   * Do the same for CIFAR100

## Running Experiments

To replicate all experiments from the paper:

1. **Run multiple experiments**:

   ```bash
   python main.py --config-name "specific_config"
   ```

   Where `"specific_config"` is one from the `conf/` directory.

2. **Plot results**:
   Add the result paths to the `paths` list in `show_results.py`, then run:

   ```bash
   python show_results.py
   ```

   Customize `custom_order` and `model_styles` as needed.

## Configuration

Training settings are defined in `conf/`:

* `dataset/` – dataset parameters
* `model/` – model architectures
* `encoder/` – input encoder configs
* `engine/` – training loop settings
* `common.yaml` – shared defaults

## Results

Experiment logs and checkpoints are saved in the Hydra output directory (default: `outputs/`).
