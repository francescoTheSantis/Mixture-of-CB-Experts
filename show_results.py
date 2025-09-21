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
                #'mnist_addition', \
                'cub', \
                'awa2',
                'awa2_incomplete',
                'cub_incomplete',
                #'cebab',
                #'celeba',
                #'cifar10',
                'cifar100',
                'dsprites_simple',
                'dsprites_complex',
                'mnist_arithmetic',
                'pendulum'
                ]

# Define a dictionary to associate marker, name, and color to each model.
# If the experiment you run does not contain a model, just remove it from the dictionary.
# If you want to add a new model, just add it to the dictionary.
marker_size = 18

# Define complexity order
models = list(reversed(['blackbox', 'cem', 'm_sym_cmr_kan', 'licem', 'l_cmr', 'dcr', 'cmr']))

# Generate colors from a colormap
cmap = plt.cm.RdYlGn 
colors = list(reversed([cmap(i) for i in np.linspace(0, 1, 5)]))

model_styles = {
    'blackbox': {'marker': 'o', 'name': 'BlackBox', 'color': colors[4], 'size': marker_size},
    'cem': {'marker': 'P', 'name': 'CEM', 'color': colors[3], 'size': marker_size},
    'm_sym_cmr_kan': {'marker': 'X', 'name': 'mCBM-sym', 'color': colors[2], 'size': marker_size},
    'l_cmr': {'marker': 's', 'name': 'mCBM-lin', 'color': colors[1], 'size': marker_size},
    'licem': {'marker': 'D', 'name': 'LICEM', 'color': colors[1], 'size': marker_size},
    'cmr': {'marker': 'v', 'name': 'mCBM-bool', 'color': colors[0], 'size': marker_size},
    'dcr': {'marker': 'h', 'name': 'DCR', 'color': colors[0], 'size': marker_size},
}


# Call the function with the desired metric and font properties
legend_font = {'size': 44}
title_font = {'size': 36, 'weight': 'bold'}
label_font = {'size': 36}
tick_font = {'size': 28}

def main():

    result_figs = "figs"
    os.makedirs(result_figs, exist_ok=True)

    ########################################################################
    ###### Visualize training/test distribution over licem's weights #######
    ########################################################################

    try:

        weights_path = "/home/fdesantis/projects/Linear-Memory-Reasoner/test/2025-09-20_21-13-39/0/logs/experiment_metrics/version_0"

        # Store the training weights
        train_w = torch.load(weights_path+'/learned_linear_coefficients_train.pt').squeeze(1)
        #iterate over all the test weights and store them in a list
        test_weights = {}
        for file in os.listdir(weights_path):
            if file.startswith('learned_linear_coefficients_test'):
                key = file.replace('learned_linear_coefficients_test_', '').replace('.pt', '')
                test_weights[key] = torch.load(os.path.join(weights_path, file)).squeeze(1)

        # read the c_names and y_names
        with open(weights_path+'/c_names.txt', 'r') as f:
            c_names = [line.strip() for line in f.readlines()]
        with open(weights_path+'/y_names.txt', 'r') as f:
            y_names = [line.strip() for line in f.readlines()]

        plot_licem_weights_distribution(
            train_w, 
            test_weights, 
            result_figs, 
            c_names, 
            y_names,
            title_font,
            label_font,
            tick_font,
            legend_font
        )

    except Exception as e:
        print(f"Error occurred while plotting LICEM weights distribution: {e}")


    ##################################################
    ###### Visualize ablation over the memory. #######
    ##################################################

    # paths = [
    #     "/home/fdesantis/projects/Linear-Memory-Reasoner/saved_outputs/blackbox/2025-09-13_20-06-02",
    #     "/home/fdesantis/projects/Linear-Memory-Reasoner/saved_outputs/cem/2025-09-13_20-06-02",
    #     "/home/fdesantis/projects/Linear-Memory-Reasoner/saved_outputs/cmr/2025-09-13_20-06-02",
    #     "/home/fdesantis/projects/Linear-Memory-Reasoner/saved_outputs/dcr/2025-09-13_20-06-06",
    #     "/home/fdesantis/projects/Linear-Memory-Reasoner/saved_outputs/l_cmr/2025-09-13_20-07-01",
    #     "/home/fdesantis/projects/Linear-Memory-Reasoner/saved_outputs/licem/2025-09-13_20-07-18",
    #     "/home/fdesantis/projects/Linear-Memory-Reasoner/saved_outputs/sym_cmr_prior/2025-09-13_20-09-06",
    #     # KANs
    #     "/home/fdesantis/projects/Linear-Memory-Reasoner/saved_outputs/sym_cmr_kan/2025-09-14_23-29-28",
    #     "/home/fdesantis/projects/Linear-Memory-Reasoner/saved_outputs/sym_cmr_kan/2025-09-14_23-29-29",
    #     "/home/fdesantis/projects/Linear-Memory-Reasoner/saved_outputs/sym_cmr_kan/2025-09-14_23-29-30",
    # ]

    paths = [
        "/home/fdesantis/projects/Linear-Memory-Reasoner/output/blackbox/2025-09-21_00-30-32",
        "/home/fdesantis/projects/Linear-Memory-Reasoner/output/cem/2025-09-21_00-30-32",
        "/home/fdesantis/projects/Linear-Memory-Reasoner/output/l_cmr/2025-09-21_00-32-28",
        "/home/fdesantis/projects/Linear-Memory-Reasoner/output/sym_cmr_kan/2025-09-21_00-30-32"
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
    
        plot_memory_ablation(performance, model_styles, title_font, label_font, tick_font, custom_order)
    except Exception as e:
        print(f"Error occurred while plotting memory ablation results: {e}")

    ##################################################
    #############  Intervention results ##############
    ##################################################

    try:

        fixed_memory={
                        'dsprites_simple': 1, 
                        'mnist_arithmetic': 4,
                        'dsprites_complex': 3,
                        'pendulum': 1
                    }

        performance, _ = get_exp_from_path(paths)
        # For the dataset for which we know the exact number of mechanisms, we select the memory size accordingly.
        filtered_exps = filter_pareto_models(
            performance, 
            fixed_memory=fixed_memory,
            custom_order=custom_order
        )

        performance = get_intervention_from_path(paths, filtered_exps=None)

        # Filter the performance dataframe to keep only the models in model_styles 
        # and datasets in custom_order.
        performance = performance[performance['model'].isin(model_styles.keys()) & \
                                performance['dataset'].isin(custom_order)]
        
        # Plot intervention results for memory ablation
        plot_intervention_memory_results(performance, 
                                        p_int=1,  # Fixed p_int value
                                        metric='accuracy', 
                                        title_font=title_font, 
                                        label_font=label_font, 
                                        tick_font=tick_font, 
                                        legend_font=legend_font,
                                        custom_order=custom_order,
                                        model_styles=model_styles,
                                        n_mechanisms=fixed_memory
                                    )

        performance = get_intervention_from_path(paths, filtered_exps=filtered_exps)

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