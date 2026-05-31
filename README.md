# Mixture of Concept Bottleneck Experts

[Paper](https://arxiv.org/abs/2602.02886)

## TL;DR
Concept Bottleneck Models (CBMs) are inherently interpretable architectures that ground predictions in human-understandable concepts. Nevertheless, existing CBMs fix the task predictor to a single, rigid expression, which limits both accuracy and interpretability. **Mixture of Concept Bottleneck Experts (M-CBEs)** generalizes CBMs along two dimensions: the **number of experts** (specialized functions mapping concepts to the task) and the **functional form** each expert takes.
We then show how this generalization allows us to instantiate two new concept-based models:
- **Linear M-CBE (Lin-M-CBE)**: a mixture of learned linear expressions.
- **Symbolic M-CBE (Sym-M-CBE)**: a mixture of expressions learned through symbolic regression constrained to a user-specified operator vocabulary.

Both models achieve a better trade-off between accuracy and interpretability compared to existing concept-based architectures.

## Method

![M-CBE method overview](assets/method.png)

The figure above shows the two instantiations (Lin-M-CBE and Sym-M-CBE) on a textual input containing questions about physics. The two models share architectural similarities. First, the input is processed by a **Concept Encoder** — which extracts interpretable physical concepts (e.g., $x_0$, $v_0$, $\beta$, $t$). Then, the **Selector** picks the most suitable function (expert) from a pool of $M$ candidate functions $f_1, \ldots, f_M$, which is then evaluated on the predicted concepts to yield the final output $y=f_m(x_0,v_0,\beta, t)$. The linear and symbolic approaches differ in how these functions are built:

**Lin-M-CBE**: each function is a parametric linear function with its own weights $\theta_m$ learned during training. The structure is fixed (linear), and only the weights vary across experts.

**Sym-M-CBE**: each function is a symbolic expression whose operators are constrained by the user. For example, the user might set $\mathcal{W}=\{+, \times, \cos, (\cdot)^2\}$ and the model will learn symbolic functions whose only operators are drawn from $\mathcal{W}$. Unlike the linear case, the model discovers both the structure of the expression and its parameters — e.g., $x_0 + v_0 \cos(\beta) \cdot t$.

## Setup

```bash
python setup_environment.py
conda activate m_cbe
```

This script creates the environment, installs dependencies, and pre-initializes Julia packages for PySR.

**Configure environment variables in** `env.py`:

* `HOME` - Path to the project root
* `DATA_PATH` - Path to load datasets
* `PROJECT_NAME` (optional) - Project identifier for logging

## Datasets

Regression datasets (MNIST-Arithm, dSprites-Exp, Pendulum, MAWPS) are generated or downloaded automatically. Classification datasets require manual setup as described below.

### AWA2

Download the Animals with Attributes 2 dataset from the [official page](https://cvml.ista.ac.at/AwA2/) and extract it under `{DATA_PATH}/AwA2/`.

### CUB-200

Download the pre-processed CUB-200 dataset following one of two options:

- **Option 1 (recommended)**: Download the pre-processed version from Koh et al.'s [CodaLab worksheet](https://worksheets.codalab.org/worksheets/0x362911581fcd4e048ddfd84f47203fd2). You need both the [original CUB bundle](https://worksheets.codalab.org/bundles/0xd013a7ba2e88481bbc07e787f73109f5) and the [CUB_preprocessed bundle](https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683).
- **Option 2**: Follow the download and preprocessing instructions in [Koh et al.'s repository](https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/).

Place the resulting files under `{DATA_PATH}/CUB_200_2011/`.

### CIFAR-10 / CIFAR-100

The image data is downloaded automatically. However, the concept annotation files must be obtained manually from the [Label-free-CBM repository](https://github.com/Trustworthy-ML-Lab/Label-free-CBM/tree/main):

* Download `cifar10_filtered.txt` and `cifar10_classes.txt` and place them in `{DATA_PATH}/cifar10/`
* Download the analogous files for CIFAR-100 and place them in `{DATA_PATH}/cifar100/`

## Running Experiments

To replicate all experiments from the paper:

1. **Run multiple experiments**:

   ```bash
   python run_sweep.py <specific_config>
   ```

   Where `<specific_config>` is one of the YAML files in the `conf/` directory.

2. **Plot results**:
   Add the result paths to the `paths` list in `show_results.py`, then run:

   ```bash
   python show_results.py
   ```

   Customize `custom_order` and `model_styles` as needed.

## Configuration

Experimental settings are defined in `conf/`:

* `dataset/` – dataset parameters
* `model/` – model architectures
* `encoder/` – input encoder configs
* `engine/` – training loop settings
* `common.yaml` – shared defaults

## Citation

If you find this work useful, please cite:

```bibtex
@article{de2026mixture,
  title={Mixture of Concept Bottleneck Experts},
  author={De Santis, Francesco and Ciravegna, Gabriele and De Felice, Giovanni and Casanova, Arianna and Giannini, Francesco and Diligenti, Michelangelo and Zarlenga, Mateo Espinosa and Barbiero, Pietro and Schneider, Johannes and Giordano, Danilo},
  journal={arXiv preprint arXiv:2602.02886},
  year={2026}
}
```