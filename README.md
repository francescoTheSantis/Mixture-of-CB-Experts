# Linear-Memory-Reasoner

Instructions to Execute the Code in This Repository:

1. Create the Conda environment:
    ```
    conda env create -f environment.yml
    ```

2. Activate the environment:
    ```
    conda activate lmr
    ```

3. Run the experiments using the configuration in `conf/sweep.yaml`:
    ```
    python main.py --config-name sweep
    ```

To modify training parameters, datasets, or models, edit the respective configurations in `conf/...` as needed before running the experiments.
