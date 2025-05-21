# Linear-Memory-Reasoner

### Setup and Execution

1. **Create the Conda environment**:
    ```
    conda env create -f environment.yml
    ```

2. **Activate the environment***:
    ```
    conda activate lmr
    ```

3. **Set the following environmental variables in** `env.py`:
    - `HOME`, Path to the project
    - `DATA_PATH`, Path to the datasets
    - (Optional) `PROJECT_NAME`, Name of the project

4. **Run the experiments using the configuration in** `conf/sweep.yaml`:
    ```
    python main.py --config-name sweep
    ```

**NOTE:** To modify training settings, datasets, or models, update the corresponding files in the `conf/` directory before running experiments.
