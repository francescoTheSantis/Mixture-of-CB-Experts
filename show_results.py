import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import warnings
import os
import yaml
import hashlib
from plot_utils import *

# I used scienceplots for the style of the plots, but you can use any other style you want.
warnings.filterwarnings("ignore")
plt.style.use(['science', 'ieee', 'no-latex'])

######### Paths to load results #########
output_path = 'output'

######### Paths to save results #########
result_figs = "results/figs"

######### Dataset and model styles #########

# Define the custom order
# If the experiment you run does not contain a dataset, just remove it from the list.
custom_order = [
    'awa2',
    'awa2_incomplete',
    'cub', 
    'cub_incomplete',
    'cifar10',
    'cifar100',
    'dsprites_simple',
    'pendulum',
    'mnist_arithmetic',
    'mawps',
]

# Regression datasets
# NOTE: update this list if you add new regression datasets
regression_datasets = [
    'mnist_arithmetic', 
    'dsprites_simple', 
    'dsprites_complex', 
    'pendulum', 
    'mawps'
]

# Define a dictionary to associate marker, name, and color to each model.
# If the experiment you run does not contain a model, just remove it from the dictionary.
# If you want to add a new model, just add it to the dictionary.
marker_size = 18

# Define complexity order
# models = list(reversed(['blackbox', 'cem', 'kan_symbolic_cbm', 'licem', 'linear_symbolic_cbm', 'dcr', 'cmr']))

MEMORY_MODELS_LIST = ['cmr', 'linear_symbolic_cbm', 'sr_symbolic_cbm', 'prior_symbolic_cbm', 'memory_cbm']

COMPLEXITY_METRICS_LIST = ['node_count', 'depth', 'visitation_length', 'total_variables', 'total_operations', 'weighted_node_count']

NUMBER_OF_CLASSES_PER_DATASET = {
    'awa2': 50,
    'awa2_incomplete': 50,
    'cub': 200,
    'cub_incomplete': 200,
    'cifar10': 10,
}

# Generate colors from a colormap
cmap = plt.cm.RdYlGn 
colors = list(reversed([cmap(i) for i in np.linspace(0, 1, 5)]))

model_styles = {
    'blackbox': {'marker': 'o', 'name': 'BlackBox', 'color': 'tab:gray', 'size': marker_size, 'fillstyle': 'none'},
    'cem': {'marker': 's', 'name': 'CEM', 'color': 'tab:red', 'size': marker_size, 'fillstyle': 'none'},

    'licem': {'marker': '^', 'name': 'LICEM', 'color': 'mediumpurple', 'size': marker_size, 'fillstyle': 'none'},
    'dcr': {'marker': 'v', 'name': 'DCR', 'color': 'tab:purple', 'size': marker_size, 'fillstyle': 'none'},

    'cbm_linear': {'marker': 'X', 'name': 'CBM', 'color': 'tab:orange', 'size': marker_size, 'fillstyle': 'none'},
    'cmr': {'marker': 'P', 'name': 'CMR', 'color': 'gold', 'size': marker_size, 'fillstyle': 'none'},

    'memory_cbm': {'marker': '*', 'name': 'MLP-M-CBE', 'color': 'tab:blue', 'size': marker_size, 'fillstyle': 'none'},
    'prior_symbolic_cbm': {'marker': '*', 'name': 'Prior-M-CBE', 'color': 'lightcyan', 'size': marker_size, 'fillstyle': 'none'},

    'kan_symbolic_cbm': {'marker': 's', 'name': 'Kan-M-CBE', 'color': 'darkmagenta', 'size': marker_size, 'fillstyle': 'none'},
    'linear_symbolic_cbm': {'marker': 'h', 'name': 'Lin-M-CBE', 'color': 'limegreen', 'size': marker_size, 'fillstyle': 'none'},
    'sr_symbolic_cbm': {'marker': 'o', 'name': 'Sym-M-CBE', 'color': 'tab:green', 'size': marker_size, 'fillstyle': 'none'},
    'bool_symbolic_cbm': {'marker': 'D', 'name': 'Bool-M-CBE', 'color': 'cyan', 'size': marker_size, 'fillstyle': 'none'},
}

models_order = [
    'blackbox', 
    'cem', 
    'licem', 
    'dcr', 
    'cbm_linear', 
    'cmr', 
    'memory_cbm', 
    'prior_symbolic_cbm', 
    'kan_symbolic_cbm', 
    'linear_symbolic_cbm', 
    'sr_symbolic_cbm',
    'bool_symbolic_cbm',
]


# Call the function with the desired metric and font properties
legend_font = {'size': 44}
title_font = {'size': 36, 'weight': 'bold'}
label_font = {'size': 36}
tick_font = {'size': 28}

# Number of mechanisms of each dataset for which those are known
fixed_memory={
    # regression datasets
    'dsprites_simple': 1, 
    'mnist_arithmetic': 4,
    'dsprites_complex': 3,
    'pendulum': 1,
    'mawps': 4,
    # classification dataset
    # None
}

def main():

    result_figs = "results/figs"
    # os.environ["RESULT_FIGS"] = result_figs
    os.makedirs(result_figs, exist_ok=True)

    paths = [
        f"{output_path}/memory_less_cls",
        f"{output_path}/memory_cls",
        f"{output_path}/memory_reg",
        f"{output_path}/memory_less_reg",
        f"{output_path}/prior_reg",
    ]

    try:
        performance = get_exp_from_path_cached(paths, cache_name='memory_ablation', output_path=output_path)

        # Count number of seeds
        seeds_count = performance.groupby(['dataset', 'model', 'memory_size'])['seed'].nunique().reset_index()
        # Save in csv file
        seeds_count.to_csv(os.path.join(result_figs, 'seeds_count_memory_ablation.csv'), index=False)

        # remove feynman datasets custom order
        refined_custom_order = [d for d in custom_order if not d.startswith('feynman')]

        # Filter the performance dataframe to keep only the models in model_styles 
        # and datasets in custom_order.
        performance = performance[performance['model'].isin(model_styles.keys()) & \
                                performance['dataset'].isin(refined_custom_order)]

        # Composed Complexity vs accuracy - Plot for each complexity metric
        complexity_metrics = ['node_count', 'depth', 'visitation_length', 'total_variables', 'total_operations', 'weighted_node_count']
        for metric in complexity_metrics:
            print(f"\nPlotting Pareto front for complexity metric: {metric}")
            plot_pareto_front(
                performance, 
                model_styles, 
                title_font, 
                label_font, 
                tick_font, 
                refined_custom_order,
                complexity_metric=metric,
                ranges=[
                    [[2, 5],[61,90]], # awa2
                    [[2, 4.5], [25, 27]], # awa2 incomplete
                    [[17, 38],[76, 101]], # cub
                    [17,50], # cub incomplete
                    [[10,23],[65,101]], # cifar10
                ],
                regression_ranges= [
                    [0.8, 1.25],
                    [0, 3.2],
                    [0.1, 9],
                    [0, 6],
                ]
            )
        
    except Exception as e:
        print(f"Error occurred while plotting memory ablation results: {e}")

    ##################################################
    #############  Intervention results ##############
    ##################################################

    try:

        # Filter the experiments in order to show only the 
        performance = get_intervention_from_path(
            paths, 
            model_styles=model_styles, 
            custom_order=custom_order, 
            apply_filter=True,
            fixed_memory=fixed_memory
        )

        ########## Intervention plots ##########
        noises = list(performance['noise'].unique())
        for noise in noises:
            plot_intervention_results(
                performance, 
                metric='accuracy', 
                # unique_noises=[noise], 
                classification_noise=[noise],
                regression_noise=[noise],
                title_font=title_font, 
                label_font=label_font, 
                tick_font=tick_font, 
                legend_font=legend_font,
                custom_order=custom_order,
                model_styles=model_styles,
                relative_accuracy=False,
                out_dir=f'{result_figs}',
            
                ranges=[
                    [[0, 5],[55,101]], # awa2
                    [[0,4],[22,27]], # awa2 incomplete
                    [[0,40],[70,101]], # cub
                    [0,45], # cub incomplete
                    [[6,21],[60,101]], # cifar10
                ],
                regression_ranges=[
                    [0, 1.5],
                    [[0,1.7],[2, 3.4]],
                    [[0, 4], [8.2,9]],
                    [0, 6.5],
                ]
            )
            
    except Exception as e:
        print(f"Error occurred while getting intervention results from path: {e}")

if __name__ == "__main__":
    main()