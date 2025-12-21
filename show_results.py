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

result_figs = "results/figs"
table_path = "results/tabs"

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
    'feynman_I_6_2',
    'feynman_I_9_18',
    'feynman_I_12_1',
    'feynman_I_13_4',
    'feynman_I_14_3',
    'feynman_I_15_10',
    'dsprites_simple',
    'pendulum',
    'dsprites_complex',
    'mnist_arithmetic',
    'mawps',
]

# Regression datasets
# NOTE: keep in mind to update this list if you add new regression datasets
regression_datasets = [
    'feynman_I_6_2',
    'feynman_I_9_18',
    'feynman_I_12_1',
    'feynman_I_13_4',
    'feynman_I_14_3',
    'feynman_I_15_10',
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

# Generate colors from a colormap
cmap = plt.cm.RdYlGn 
colors = list(reversed([cmap(i) for i in np.linspace(0, 1, 5)]))

model_styles = {
    'blackbox': {'marker': 'o', 'name': 'BlackBox', 'color': "black", 'size': marker_size},
    'cbm_linear': {'marker': '^', 'name': 'CBM', 'color': 'tab:orange', 'size': marker_size},
    'cem': {'marker': 'P', 'name': 'CEM', 'color': 'tab:red', 'size': marker_size},
    'kan_symbolic_cbm': {'marker': 'X', 'name': 'Kan-Sym-CBM', 'color': 'tab:green', 'size': marker_size},
    'linear_symbolic_cbm': {'marker': 's', 'name': 'Lin-Sym-CBM', 'color': 'tab:orange', 'size': marker_size},
    'prior_symbolic_cbm': {'marker': '*', 'name': 'Prior-Sym-CBM', 'color': 'tab:cyan', 'size': marker_size},
    'sr_symbolic_cbm': {'marker': 'o', 'name': 'SR-Sym-CBM', 'color': 'tab:blue', 'size': marker_size},
    'licem': {'marker': 'D', 'name': 'LICEM', 'color': 'tab:brown', 'size': marker_size},
    'cmr': {'marker': 'v', 'name': 'CMR', 'color': 'tab:pink', 'size': marker_size},
    'dcr': {'marker': 'h', 'name': 'DCR', 'color': 'tab:purple', 'size': marker_size},
}

# Call the function with the desired metric and font properties
legend_font = {'size': 44}
title_font = {'size': 36, 'weight': 'bold'}
label_font = {'size': 36}
tick_font = {'size': 28}

# Number of mechanisms of each dataset for which those are known
fixed_memory={
                'dsprites_simple': 1, 
                'mnist_arithmetic': 4,
                'dsprites_complex': 3,
                'pendulum': 1,
                'mawps': 4,
            }

def main():

    result_figs = "results/figs"
    # os.environ["RESULT_FIGS"] = result_figs
    os.makedirs(result_figs, exist_ok=True)

    table_path = "results/tabs"
    # os.environ["TABLE_PATH"] = table_path
    os.makedirs(table_path, exist_ok=True)

    #######################################################################
    ###### Visualize the results of the symbolic regression ablation ######
    #######################################################################

    # 1. show the accuracy results of symbolic regression tasks in tabular format
    # 2. plot the intervention curves
    # 3. compute the similarity between the learned expressions and the ground truth ones

    paths = [
        # "output/sr_ablation",
        # "output/prior_reg",
    ]
 
    try:
        performance, _ = get_exp_from_path(paths)
        # Filter the performance dataframe to keep only the models in model_styles 
        # and datasets in custom_order.
        performance = performance[performance['model'].isin(model_styles.keys()) & \
                                performance['dataset'].isin(custom_order)]
        
        show_symbolic_regression_results(performance, custom_order, table_path)

        # Now plot intervention results with noise=0.0
        performance = get_intervention_from_path(paths, filtered_exps=None)

        # Filter the performance dataframe to keep only the models in model_styles 
        # and datasets in custom_order.
        performance = performance[performance['model'].isin(model_styles.keys()) & \
                                performance['dataset'].isin(custom_order)]

        # Plot intervention results with noise=0.0
        plot_intervention_results(performance, 
                                    metric='accuracy', 
                                    unique_noises=[0.0], 
                                    title_font=title_font, 
                                    label_font=label_font, 
                                    tick_font=tick_font, 
                                    legend_font=legend_font,
                                    custom_order=custom_order,
                                    model_styles=model_styles,
                                    relative_accuracy=False,
                                    out_dir=f'{result_figs}/sr_ablation')

        # Compute equation complexity metrics
        print("\nComputing equation complexity metrics...")
        complexity_results = compute_equation_complexity_for_sr_ablation(paths, n_samples=1)
        complexity_csv_path = os.path.join(table_path, 'sr_ablation', 'complexity_metrics.csv')
        complexity_results.to_csv(complexity_csv_path, index=False)
        print(f"Complexity metrics saved to {complexity_csv_path}")
        print(f"Complexity results summary:\n{complexity_results}")

        # Compare equations with prior model
        print("\nComparing learned equations with prior model...")
        equation_comparison = compare_equations_with_prior(paths)
        equation_comparison_csv_path = os.path.join(table_path, 'sr_ablation', 'equation_comparison.csv')
        equation_comparison.to_csv(equation_comparison_csv_path, index=False)
        print(f"Equation comparison saved to {equation_comparison_csv_path}")
        if len(equation_comparison) > 0:
            print(f"Compared {len(equation_comparison)} equations across {equation_comparison['dataset'].nunique()} datasets")
    
        # Compute Tree Edit Distance (TED) metrics
        print("\nComputing Tree Edit Distance (TED) metrics...")
        ted_results = compute_ted_metrics_for_sr_ablation(paths)
        ted_csv_path = os.path.join(table_path, 'sr_ablation', 'ted_metrics.csv')
        ted_results.to_csv(ted_csv_path, index=False)
        print(f"TED metrics saved to {ted_csv_path}")
        print(f"TED results summary:\n{ted_results.groupby(['dataset', 'model'])['avg_ted'].agg(['mean', 'std', 'count'])}")

    except Exception as e:
        print(f"Error occurred while plotting Symbolic Regression ablation results: {e}")

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

    ##################################################
    ######### Visualize performance results ##########
    ##################################################

    paths = [
        "output/memory_less_cls",
        "output/memory_cls",
        "output/memory_reg",
        "output/memory_less_reg"
    ]

    try:
        performance, _ = get_exp_from_path(paths)

        # Count number of seeds
        seeds_count = performance.groupby(['dataset', 'model', 'memory_size'])['seed'].nunique().reset_index()
        # Save in csv file
        seeds_count.to_csv(os.path.join(result_figs, 'seeds_count_memory_ablation.csv'), index=False)

        # Filter the performance dataframe to keep only the models in model_styles 
        # and datasets in custom_order.
        performance = performance[performance['model'].isin(model_styles.keys()) & \
                                performance['dataset'].isin(custom_order)]

        plot_memory_ablation(performance, model_styles, title_font, label_font, tick_font, custom_order)

        # Plot the models in the pareto front
        plot_pareto_front(performance, 
                        model_styles, 
                        title_font, 
                        label_font, 
                        tick_font, 
                        custom_order,
        )
        
    except Exception as e:
        print(f"Error occurred while plotting memory ablation results: {e}")

    ##################################################
    #############  Intervention results ##############
    ##################################################

    try:

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
                                        n_mechanisms=fixed_memory,
                                    )
        
        plot_intervention_memory_pareto_results(performance, 
                                    p_int=1,  
                                    metric='accuracy', 
                                    title_font=title_font, 
                                    label_font=label_font, 
                                    tick_font=tick_font, 
                                    legend_font=legend_font,
                                    custom_order=custom_order,
                                    model_styles=model_styles,
                                    n_mechanisms=fixed_memory,
        )

        performance = get_intervention_from_path(paths, filtered_exps=filtered_exps)

        # Filter the performance dataframe to keep only the models in model_styles 
        # and datasets in custom_order.
        performance = performance[performance['model'].isin(model_styles.keys()) & \
                                performance['dataset'].isin(custom_order)]
        
        ########## Intervention plots ##########
        noises = list(performance['noise'].unique())
        for noise in noises:
            plot_intervention_results(
                performance, 
                metric='accuracy', 
                unique_noises=[noise], 
                title_font=title_font, 
                label_font=label_font, 
                tick_font=tick_font, 
                legend_font=legend_font,
                custom_order=custom_order,
                model_styles=model_styles,
                relative_accuracy=False,
            )
            
            plot_intervention_results(
                performance, 
                metric='accuracy', 
                unique_noises=[noise], 
                title_font=title_font, 
                label_font=label_font, 
                tick_font=tick_font, 
                legend_font=legend_font,
                custom_order=custom_order,
                model_styles=model_styles,
                relative_accuracy=True,
            )
            
    except Exception as e:
        print(f"Error occurred while getting intervention results from path: {e}")

if __name__ == "__main__":
    main()