import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import warnings
import os
import yaml
from plot_utils import *

# I used scienceplots for the style of the plots, but you can use any other style you want.
warnings.filterwarnings("ignore")
plt.style.use(['science', 'ieee', 'no-latex'])

######### Dataset and model styles #########

# Define the custom order
# If the experiment you run does not contain a dataset, just remove it from the list.
custom_order = [#'xor', \
                #'dot', \
                #'checkmark', \
                #'trigonometry', \
                'mnist_addition', \
                'cub', \
                'awa2',
                'awa2_incomplete',
                'cub_incomplete',
                #'cebab',
                #'celeba',
                #'cifar10',
                #'cifar100',
                'dsprites_simple',
                'mnist_arithmetic',
                ]

# Define a dictionary to associate marker, name, and color to each model.
# If the experiment you run does not contain a model, just remove it from the dictionary.
# If you want to add a new model, just add it to the dictionary.
marker_size = 14
model_styles = {
    'cem': {'marker': 'P', 'name': 'CEM', 'color': 'tab:blue', 'size': marker_size},
    'cbm_linear': {'marker': '*', 'name': 'CBM', 'color': 'tab:orange', 'size': marker_size},
    #'cbm_mlp': {'marker': '^', 'name': 'CBM+MLP', 'color': 'tab:red', 'size': marker_size},
    'blackbox': {'marker': 'o', 'name': 'BlackBox', 'color': 'tab:purple', 'size': marker_size},
    'cmr': {'marker': 'v', 'name': 'CMR', 'color': 'tab:pink', 'size': marker_size},
    'dcr': {'marker': 'h', 'name': 'DCR', 'color': 'tab:gray', 'size': marker_size},
    'licem': {'marker': 'D', 'name': 'LICEM', 'color': 'tab:cyan', 'size': marker_size},
    'l_cmr': {'marker': 's', 'name': 'L-CMR', 'color': 'tab:green', 'size': marker_size},
    'm_sym_cmr_prior': {'marker': 'X', 'name': 'Sym-CMR-With-Prior', 'color': 'tab:brown', 'size': marker_size},
    'm_sym_cmr_kan': {'marker': 'X', 'name': 'Sym-CMR', 'color': 'tab:olive', 'size': marker_size},
}

# Call the function with the desired metric and font properties
legend_font = {'size': 44}
title_font = {'size': 36, 'weight': 'bold'}
label_font = {'size': 36}
tick_font = {'size': 28}

def main():

    result_figs = "figs"
    os.makedirs(result_figs, exist_ok=True)

    ##################################################
    ###### Visualize ablation over the memory. #######
    ##################################################

    paths = [
        "/home/fdesantis/projects/Linear-Memory-Reasoner/output/blackbox/2025-09-12_12-32-24",
        "/home/fdesantis/projects/Linear-Memory-Reasoner/output/cbm_linear/2025-09-12_12-32-24",
        "/home/fdesantis/projects/Linear-Memory-Reasoner/output/cem/2025-09-12_12-33-26",
        "/home/fdesantis/projects/Linear-Memory-Reasoner/output/cmr/2025-09-12_12-35-02",
        "/home/fdesantis/projects/Linear-Memory-Reasoner/output/dcr/2025-09-12_12-35-38",
        "/home/fdesantis/projects/Linear-Memory-Reasoner/output/licem/2025-09-12_12-38-31",
        "/home/fdesantis/projects/Linear-Memory-Reasoner/output/sym_cmr_kan/2025-09-12_12-40-40",
        "/home/fdesantis/projects/Linear-Memory-Reasoner/output/l_cmr/2025-09-12_12-38-31"
    ]

    try:
        performance, _ = get_exp_from_path(paths)

        # Count number of seeds
        num_seeds = performance['seed'].unique().max()
        print(f"Number of unique seeds: {num_seeds}")

        # Filter the performance dataframe to keep only the models in model_styles 
        # and datasets in custom_order.
        performance = performance[performance['model'].isin(model_styles.keys()) & \
                                performance['dataset'].isin(custom_order)]
    
        plot_memory_ablation(performance, model_styles, title_font, label_font, tick_font)
    except Exception as e:
        print(f"Error occurred while plotting memory ablation results: {e}")

    ##################################################
    #############  Intervention results ##############
    ##################################################

    try:
        performance, _ = get_exp_from_path(paths)
        # For the dataset for which we know the exact number of mechanisms, we select the memory size accordingly.
        filtered_exps = filter_pareto_models(performance, fixed_memory={'dsprites_simple': 1, 'mnist_arithmetic': 4})
        performance = get_intervention_from_path(paths, filtered_exps)

        # Filter the performance dataframe to keep only the models in model_styles 
        # and datasets in custom_order.
        performance = performance[performance['model'].isin(model_styles.keys()) & \
                                performance['dataset'].isin(custom_order)]

        # Count number of seeds
        num_seeds = performance['seed'].unique().max()
        print(f"Number of unique seeds: {num_seeds}")

        # Filter the performance dataframe to keep only the models in model_styles 
        # and datasets in custom_order.
        performance = performance[performance['model'].isin(model_styles.keys()) & \
                                performance['dataset'].isin(custom_order)]

        ########## Intervention plots ##########
        noises = list(performance['noise'].unique())
        for noise in noises:
            plot_intervention_results(performance, 
                                        metric='accuracy', 
                                        unique_noises=[noise], 
                                        title_font=title_font, 
                                        label_font=label_font, 
                                        tick_font=tick_font, 
                                        legend_font=legend_font,
                                        custom_order=custom_order,
                                        model_styles=model_styles,
                                        relative_accuracy=False)
            
            plot_intervention_results(performance, 
                                        metric='accuracy', 
                                        unique_noises=[noise], 
                                        title_font=title_font, 
                                        label_font=label_font, 
                                        tick_font=tick_font, 
                                        legend_font=legend_font,
                                        custom_order=custom_order,
                                        model_styles=model_styles,
                                        relative_accuracy=True)
    except Exception as e:
        print(f"Error occurred while getting intervention results from path: {e}")


    ############################################################
    ###### Visualize the results of concept size ablation ######
    ############################################################

    paths = [
        "",
    ]

    try:
        performance, _ = get_exp_from_path(paths)
        # Plot the results on the concept size ablation
        plot_concept_size_ablation(performance, model_styles, title_font, label_font, tick_font)
    except Exception as e:
        print(f"Error occurred while plotting memory ablation results: {e}")


if __name__ == "__main__":
    main()