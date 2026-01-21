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
    # 'cifar100',
    'feynman_I_6_2',
    'feynman_I_9_18',
    'feynman_I_12_1',
    'feynman_I_13_4',
    'feynman_I_14_3',
    'feynman_I_15_10',
    'dsprites_simple',
    'pendulum',
    # 'dsprites_complex',
    'mnist_arithmetic',
    'mawps',
]

# Regression datasets
# NOTE: update this list if you add new regression datasets
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

MEMORY_MODELS_LIST = ['cmr', 'linear_symbolic_cbm', 'sr_symbolic_cbm', 'prior_symbolic_cbm', 'memory_cbm']

COMPLEXITY_METRICS_LIST = ['node_count', 'depth', 'visitation_length', 'total_variables', 'total_operations', 'weighted_node_count']

# Generate colors from a colormap
cmap = plt.cm.RdYlGn 
colors = list(reversed([cmap(i) for i in np.linspace(0, 1, 5)]))

model_styles = {
    # =========================
    # BASELINES (non-verifiable)
    # =========================
    'blackbox': {'marker': 'o', 'name': 'BlackBox', 'color': 'tab:gray', 'size': marker_size, 'fillstyle': 'none'},
    'cem': {'marker': 's', 'name': 'CEM', 'color': 'tab:red', 'size': marker_size, 'fillstyle': 'none'},
    'licem': {'marker': 'D', 'name': 'LICEM', 'color': 'tab:brown', 'size': marker_size, 'fillstyle': 'none'},
    'dcr': {'marker': 'v', 'name': 'DCR', 'color': 'tab:purple', 'size': marker_size, 'fillstyle': 'none'},
    # =========================
    # BASELINE but VERIFIABLE
    # =========================
    'cmr': {'marker': 'P', 'name': 'CMR', 'color': 'yellow', 'size': marker_size, 'fillstyle': 'none'},
    'cbm_linear': {'marker': '^', 'name': 'CBM', 'color': 'tab:orange', 'size': marker_size, 'fillstyle': 'none'},
    # =========================
    # PROPOSED MODELS (verifiable)
    # =========================
    'memory_cbm': {'marker': 'X', 'name': 'MLP-Mem-CBM', 'color': 'tab:blue', 'size': marker_size, 'fillstyle': 'none'},
    'prior_symbolic_cbm': {'marker': '*', 'name': 'Prior-Mem-CBM', 'color': 'tab:cyan', 'size': marker_size, 'fillstyle': 'none'},
    'sr_symbolic_cbm': {'marker': 'h', 'name': 'Sym-Mem-CBM', 'color': 'tab:green', 'size': marker_size, 'fillstyle': 'none'},
    'linear_symbolic_cbm': {'marker': '8', 'name': 'Lin-Mem-CBM', 'color': 'tab:olive', 'size': marker_size, 'fillstyle': 'none'},
    'kan_symbolic_cbm': {'marker': 'p', 'name': 'Kan-Mem-CBM', 'color': 'tab:teal', 'size': marker_size, 'fillstyle': 'none'},
}


# Call the function with the desired metric and font properties
legend_font = {'size': 44}
title_font = {'size': 36, 'weight': 'bold'}
label_font = {'size': 36}
tick_font = {'size': 28}

# Number of mechanisms of each dataset for which those are known
fixed_memory={
    # regression datasets
    'feynman_I_6_2': 1,
    'feynman_I_9_18': 1,
    'feynman_I_12_1': 1,
    'feynman_I_13_4': 1,
    'feynman_I_14_3': 1,
    'feynman_I_15_10': 1,
    'dsprites_simple': 1, 
    'mnist_arithmetic': 4,
    'dsprites_complex': 3,
    'pendulum': 1,
    'mawps': 4,
    # classification datasets
    'awa2': 1,
    'awa2_incomplete': 1,
    'cub': 1, 
    'cub_incomplete': 1,
    'cifar10': 1,
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

    paths = [
        f"{output_path}/sr_ablation",
        f"{output_path}/prior_reg",
    ]
 
    try:
        performance = get_exp_from_path_cached(paths, cache_name='sr_ablation', output_path=output_path)
        # Filter the performance dataframe to keep only the models in model_styles 
        # and datasets in custom_order.
        performance = performance[performance['model'].isin(model_styles.keys()) & performance['dataset'].isin(custom_order)]
        
        show_symbolic_regression_results(performance, custom_order, table_path)

        # Compute equation complexity metrics
        print("\nComputing equation complexity metrics...")

        # Include all complexity metrics in the results
        complexity_cols = ['dataset', 'model', 'seed']
        complexity_metrics = ['node_count', 'depth', 'visitation_length', 'total_variables', 'total_operations', 'weighted_node_count']
        for metric in complexity_metrics:
            col_name = f'complexity_{metric}'
            if col_name in performance.columns:
                complexity_cols.append(col_name)
        
        complexity_results = performance[complexity_cols].copy()
        complexity_csv_path = os.path.join(table_path, 'sr_ablation', 'complexity_metrics.csv')
        complexity_results.to_csv(complexity_csv_path, index=False)
        print(f"Complexity metrics saved to {complexity_csv_path}")
        print(f"Complexity results summary:\n{complexity_results.describe()}")
        
        # Generate LaTeX tables for SR ablation with all complexity metrics
        print("\nGenerating SR ablation performance tables with all complexity metrics...")
        generate_performance_tables(
            performance,
            [d for d in custom_order if d in performance['dataset'].unique()],
            model_styles,
            regression_datasets,
            os.path.join(table_path, 'sr_ablation')
        )

        # Generate concept metrics table for SR ablation
        print("\nGenerating SR ablation concept metrics table...")
        generate_concept_metrics_table(
            performance,
            [d for d in custom_order if d in performance['dataset'].unique()],
            model_styles,
            regression_datasets,
            os.path.join(table_path, 'sr_ablation')
        )

        # Compute Tree Edit Distance (TED) metrics
        print("\nComputing Tree Edit Distance (TED) metrics...")
        ted_results, equations_df = compute_ted_metrics(paths)
        ted_csv_path = os.path.join(table_path, 'sr_ablation', 'ted_metrics.csv')
        ted_results.to_csv(ted_csv_path, index=False)
        print(f"TED metrics saved to {ted_csv_path}")
        print(f"TED results summary:\n{ted_results.groupby(['dataset', 'model'])['avg_ted'].agg(['mean', 'std', 'count'])}")
        equations_csv_path = os.path.join(table_path, 'sr_ablation', 'equations.csv')
        equations_df.to_csv(equations_csv_path, index=False)
        print(f"Learned equations saved to {equations_csv_path}")

        # Now plot intervention results with noise=0.0
        performance = get_intervention_from_path(paths)

        # Filter the performance dataframe to keep only the models in model_styles 
        # and datasets in custom_order.
        performance = performance[performance['model'].isin(model_styles.keys()) & \
                                performance['dataset'].isin(custom_order)]

        # Plot intervention results
        for noise in performance['noise'].unique():
            plot_intervention_results(performance, 
                                        metric='accuracy', 
                                        unique_noises=[noise], 
                                        title_font=title_font, 
                                        label_font=label_font, 
                                        tick_font=tick_font, 
                                        legend_font=legend_font,
                                        custom_order=custom_order,
                                        model_styles=model_styles,
                                        relative_accuracy=False,
                                        out_dir=f'{result_figs}/sr_ablation')

    except Exception as e:
        print(f"Error occurred while plotting Symbolic Regression ablation results: {e}")

    ############################################################
    ###### Visualize the results of concept size ablation ######
    ############################################################

    paths = [
        "",
    ]

    try:
        performance = get_exp_from_path_cached(paths, cache_name='concept_size_ablation', output_path=output_path)
        # Plot the results on the concept size ablation
        plot_concept_size_ablation(performance, model_styles, title_font, label_font, tick_font)
    except Exception as e:
        print(f"Error occurred while plotting concept size ablation results: {e}")    

    ##################################################
    ######### Visualize performance results ##########
    ##################################################

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

        # Memory vs accuracy 
        plot_memory_ablation(
            performance, 
            model_styles, 
            title_font, 
            label_font, 
            tick_font, 
            refined_custom_order
        )

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
                    [0,100], # awa2
                    [0,100], # awa2 incomplete
                    [0,100], # cub
                    [0,100], # cub incomplete
                    [0,100], # cifar10
                ]
            )
        
        # Generate performance tables for classification and regression datasets
        print("\nGenerating performance tables with all complexity metrics...")
        generate_performance_tables(
            performance,
            refined_custom_order,
            model_styles,
            regression_datasets,
            table_path
        )

        # Generate concept metrics table
        print("\nGenerating concept metrics table...")
        generate_concept_metrics_table(
            performance,
            refined_custom_order,
            model_styles,
            regression_datasets,
            table_path
        )

        # Compute TED metric for regression datasets (datasets for which we know the ground truth expressions/mechanisms)
        ted_paths = [
            f"{output_path}/memory_reg",
            f"{output_path}/memory_less_reg",
            f"{output_path}/prior_reg",
        ]
        print("\nComputing Tree Edit Distance (TED) metrics...")
        ted_results, _ = compute_ted_metrics(ted_paths)
        # Filter prior and models which did not match the number of true mechanisms
        ted_results = ted_results[(ted_results['memory_size']==ted_results['n_true']) & (ted_results['model']!='prior_symbolic_cbm')]
        ted_csv_path = os.path.join(table_path, 'ted_metrics.csv')
        ted_results.to_csv(ted_csv_path, index=False)
        ted_tex_path = os.path.join(table_path, 'ted_metrics.txt')
        csv_to_table(
            path=ted_tex_path,
            df=ted_results, 
            dataset_col='dataset',
            model_col='model',
            mean_col='avg_ted',
            std_col='se_ted',
            custom_order=refined_custom_order,
            model_styles=model_styles,
        )
        print(f"TED metrics saved to {ted_csv_path}")
        print(f"TED results summary:\n{ted_results.groupby(['dataset', 'model'])['avg_ted'].agg(['mean', 'std', 'count'])}")

    except Exception as e:
        print(f"Error occurred while plotting memory ablation results: {e}")


    ##################################################
    ############## Global verifiability ##############
    ##################################################

    # global_verifiability_plot(
    #     paths, 
    #     dataset='cub',  # Change dataset here
    #     memory_size=4,     # Change memory size here
    #     label_font=label_font, 
    #     tick_font=tick_font, 
    #     legend_font=legend_font
    # )


    ##################################################
    #############  Intervention results ##############
    ##################################################

    try:

        performance = get_exp_from_path_cached(paths, cache_name='memory_ablation', output_path=output_path)

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
                regression_noise=[0.0],
                title_font=title_font, 
                label_font=label_font, 
                tick_font=tick_font, 
                legend_font=legend_font,
                custom_order=custom_order,
                model_styles=model_styles,
                relative_accuracy=False,
                out_dir=f'{result_figs}',
                ranges=[
                    [0,100], # awa2
                    [0,100], # awa2 incomplete
                    [0,100], # cub
                    [0,100], # cub incomplete
                    [0,100], # cifar10
                ]
            )
            
    except Exception as e:
        print(f"Error occurred while getting intervention results from path: {e}")


    ##################################################
    ######### Adaptability Experiment Results ########
    ##################################################
    
    print("\n" + "="*70)
    print("ADAPTABILITY EXPERIMENT RESULTS")
    print("="*70)
    
    adaptability_paths = [
        f"{output_path}/adaptability_experiment",
    ]
    
    # Define constraint configurations from the experiment config
    constraint_configs = ['simple', 'medium', 'complex']
    
    try:
        performance = get_adaptability_exp_from_path(adaptability_paths, constraint_configs)
        
        if not performance.empty:
            # Get unique datasets from the results
            datasets_in_results = sorted(performance['dataset'].unique())
            
            # Filter to only include datasets that are in custom_order
            datasets_to_show = [d for d in custom_order if d in datasets_in_results]
            
            print(f"\nFound {len(performance)} results across {len(datasets_to_show)} datasets")
            print(f"Datasets: {datasets_to_show}")
            print(f"Model configurations: {sorted(performance['model'].unique())}")
            
            # Generate adaptability table
            adaptability_output_path = os.path.join(table_path, 'adaptability_experiment')
            result_table = generate_adaptability_table(
                performance, 
                datasets_to_show, 
                adaptability_output_path,
                regression_datasets
            )
            
            print(f"\n✓ Adaptability experiment results processed successfully!")
            print(f"\nTable preview:")
            print(result_table.to_string(index=False))
        else:
            print("No adaptability experiment results found.")
            
    except Exception as e:
        print(f"Error occurred while processing adaptability experiment results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()