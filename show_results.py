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
    # 'feynman_I_6_2',
    # 'feynman_I_9_18',
    # 'feynman_I_12_1',
    # 'feynman_I_13_4',
    # 'feynman_I_14_3',
    # 'feynman_I_15_10',
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

    # 'kan_symbolic_cbm': {'marker': 's', 'name': 'Kan-M-CBE', 'color': 'darkmagenta', 'size': marker_size, 'fillstyle': 'none'},
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
    # 'awa2': 1,
    # 'awa2_incomplete': 3,
    # 'cub': 1, 
    # 'cub_incomplete': 2,
    # 'cifar10': 2,
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

        # Generate TED metrics LaTeX table
        print("\nGenerating TED metrics LaTeX table...")
        generate_ted_metrics_table(
            ted_results,
            [d for d in custom_order if d in ted_results['dataset'].unique()],
            model_styles,
            os.path.join(table_path, 'sr_ablation')
        )

        # Generate complexity metrics LaTeX tables for each metric
        print("\nGenerating complexity metrics LaTeX tables...")
        complexity_metrics_list = ['node_count', 'depth', 'visitation_length', 'total_variables', 'total_operations', 'weighted_node_count']
        for metric in complexity_metrics_list:
            generate_complexity_metrics_table(
                complexity_results,
                [d for d in custom_order if d in complexity_results['dataset'].unique()],
                model_styles,
                os.path.join(table_path, 'sr_ablation'),
                metric=metric
            )

        # Now plot intervention results with noise=0.0
        performance = get_intervention_from_path(paths)

        # Filter the performance dataframe to keep only the models in model_styles 
        # and datasets in custom_order.
        performance = performance[performance['model'].isin(model_styles.keys()) & \
                                performance['dataset'].isin(custom_order)]

        # Plot intervention results
        for noise in performance['noise'].unique():
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
                out_dir=f'{result_figs}/sr_ablation'
            )
        
        # Generate SR ablation intervention table (MAE at p_int=0.0)
        print("\nGenerating SR ablation intervention table...")
        sr_ablation_intervention_path = os.path.join(table_path, 'sr_ablation')
        
        # Filter datasets to only include MAWPS
        sr_datasets = ['mawps']
        
        # Filter models - you can customize this list
        sr_models = ['licem', 'linear_symbolic_cbm', 'sr_symbolic_cbm', "cem"]
        
        # Specify noise levels to include
        sr_noise_levels = [0.0, 0.3]
        
        sr_intervention_results = generate_sr_ablation_intervention_table(
            performance,
            custom_order,
            model_styles,
            sr_ablation_intervention_path,
            models_to_include=sr_models,
            datasets_to_include=sr_datasets,
            noise_levels_to_include=sr_noise_levels
        )
        
        if sr_intervention_results is not None:
            print(f"✓ SR ablation intervention table generated successfully!")
        else:
            print("Failed to generate SR ablation intervention table")

    except Exception as e:
        print(f"Error occurred while plotting Symbolic Regression ablation results: {e}")

    ############################################################
    ###### Visualize the results of concept size ablation ######
    ############################################################

    paths = [
        "output/concept_size_ablation",
    ]

    try:
        performance = get_exp_from_path_cached(paths, cache_name='concept_size_ablation', output_path=output_path)
        # Plot the results on the concept size ablation
        plot_concept_size_ablation(performance, model_styles, title_font, label_font, tick_font, custom_order)
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

    global_verifiability_plot(
        paths, 
        dataset='cub_incomplete',  # Change dataset here
        memory_size=4,     # Change memory size here
        label_font=label_font, 
        tick_font=tick_font, 
        legend_font=legend_font
    )


    ##################################################
    #############  Intervention results ##############
    ##################################################

    try:

        # performance = get_exp_from_path_cached(paths, cache_name='memory_ablation', output_path=output_path)

        # # Take the rows in the dataset when model=linear_symbolic_cbm and memory_size=1
        # cbm = performance[(performance['model']=='linear_symbolic_cbm') & (performance['memory_size']==1)]
        # cbm['model'] = 'cbm_linear'
        # performance = pd.concat([performance, cbm], ignore_index=True)

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
                regression_noise=[0.1],
                title_font=title_font, 
                label_font=label_font, 
                tick_font=tick_font, 
                legend_font=legend_font,
                custom_order=custom_order,
                model_styles=model_styles,
                relative_accuracy=False,
                out_dir=f'{result_figs}',
                ranges=[
                    [[0, 5],[70,80]], # awa2
                    [[0,4],[22,27]], # awa2 incomplete
                    [[0,40],[70,101]], # cub
                    [0,45], # cub incomplete
                    [[6,21],[80,95]], # cifar10
                ],
                regression_ranges=[
                    [0, 1.5],
                    [[0,1.7],[2, 3.4]],
                    [[0, 4], [8.2,9]],
                    [0, 6.5],
                ]
            )
        
        ########## Intervention Delta Table ##########
        print("\n" + "="*70)
        print("GENERATING INTERVENTION DELTA TABLE")
        print("="*70)
        
        # Generate intervention delta table for noise=0.0
        intervention_delta_path = os.path.join(table_path, 'intervention_delta')
        os.makedirs(intervention_delta_path, exist_ok=True)
        
        # Remove feynman datasets from custom_order for this table
        refined_custom_order = [d for d in custom_order if not d.startswith('feynman')]
        
        intervention_delta_results = generate_intervention_delta_table(
            performance,
            refined_custom_order,
            model_styles,
            regression_datasets,
            intervention_delta_path,
            noise_level=0.0
        )
        
        if intervention_delta_results is not None:
            print(f"\n✓ Intervention delta table generated successfully!")
            print(f"\nSummary statistics:")
            print(intervention_delta_results.groupby('model')['delta_mean'].agg(['mean', 'std', 'count']))
        else:
            print("Failed to generate intervention delta table")
            
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
            
            # Generate memory-complexity tables (one per dataset)
            print("\nGenerating memory-complexity tables for each dataset...")
            memory_complexity_results = generate_adaptability_memory_complexity_tables(
                performance,
                datasets_to_show,
                adaptability_output_path,
                base_model='sr_symbolic_cbm'
            )
            
            if memory_complexity_results:
                print(f"\n✓ Generated {len(memory_complexity_results)} memory-complexity tables!")
                for dataset, df in memory_complexity_results.items():
                    print(f"  - {dataset}: {len(df)} constraint sets")
            else:
                print("No memory-complexity tables generated")
        else:
            print("No adaptability experiment results found.")
            
    except Exception as e:
        print(f"Error occurred while processing adaptability experiment results: {e}")
        import traceback
        traceback.print_exc()


    ##################################################
    ######## Global Interpretability Table ###########
    ##################################################

    print("\n" + "="*70)
    print("GLOBAL INTERPRETABILITY TABLE")
    print("="*70)

    try:
        gi_results_df = collect_global_interpretability_results(f"{output_path}/global_interpretability")

        if not gi_results_df.empty:
            print(f"Collected {len(gi_results_df)} rows from global interpretability experiments.")
            print(f"Models: {gi_results_df['model'].unique()}")
            print(f"Datasets: {gi_results_df['dataset'].unique()}")
            print(f"Seeds: {gi_results_df['seed'].unique()}")

            gi_output_path = os.path.join(table_path, 'global_interpretability')
            os.makedirs(gi_output_path, exist_ok=True)

            # Save aggregated raw results
            gi_results_df.to_csv(os.path.join(gi_output_path, 'global_interpretability_all.csv'), index=False)

            # Determine classification vs regression datasets present
            all_gi_datasets = sorted(gi_results_df['dataset'].unique())
            cls_datasets = [d for d in all_gi_datasets if d not in regression_datasets]
            reg_datasets = [d for d in all_gi_datasets if d in regression_datasets]

            # Generate range tables (classification + regression)
            for ds_filter, ds_label, ds_caption in [
                (cls_datasets, '_cls', ' Classification datasets.'),
                (reg_datasets, '_reg', ' Regression datasets.'),
            ]:
                if not ds_filter:
                    continue
                latex_table = build_global_interpretability_table(
                    gi_results_df, n_display_concepts=3,
                    dataset_filter=ds_filter, label_suffix=ds_label, caption_suffix=ds_caption
                )
                tex_path = os.path.join(gi_output_path, f'global_interpretability{ds_label}.tex')
                with open(tex_path, 'w') as f:
                    f.write(latex_table)
                print(f"LaTeX table saved to: {tex_path}")

            # Generate violation tables (classification + regression)
            if 'n_violations' in gi_results_df.columns:
                for mode, suffix in [('count', 'count'), ('percentage', 'pct')]:
                    for ds_filter, ds_label, ds_caption in [
                        (cls_datasets, '_cls', ' Classification datasets.'),
                        (reg_datasets, '_reg', ' Regression datasets.'),
                    ]:
                        if not ds_filter:
                            continue
                        table = build_global_interpretability_violations_table(
                            gi_results_df, n_display_concepts=3, mode=mode,
                            dataset_filter=ds_filter, label_suffix=ds_label, caption_suffix=ds_caption
                        )
                        tex_path = os.path.join(gi_output_path, f'global_interpretability_violations_{suffix}{ds_label}.tex')
                        with open(tex_path, 'w') as f:
                            f.write(table)
                        print(f"Violation {suffix} table saved to: {tex_path}")
            else:
                print("No violation data in results (n_violations column missing). Re-run experiments to generate.")
        else:
            print("No global interpretability results found.")
    except Exception as e:
        print(f"Error occurred while processing global interpretability results: {e}")
        import traceback
        traceback.print_exc()


    ##################################################
    ######### Expressions Intervention ###############
    ##################################################

    print("\n" + "="*70)
    print("EXPRESSIONS INTERVENTION EXPERIMENT")
    print("="*70)

    expr_interv_path = os.path.join(output_path, 'expressions_intervention')
    expr_interv_output = os.path.join(table_path, 'expressions_intervention')
    os.makedirs(expr_interv_output, exist_ok=True)

    try:
        cached_csv = os.path.join(output_path, 'cached_results', 'expressions_intervention_results.csv')
        if os.path.exists(cached_csv):
            print(f"Loading cached results from {cached_csv}")
            expr_df = pd.read_csv(cached_csv)
        else:
            expr_df = collect_expressions_intervention_results(expr_interv_path)
            if not expr_df.empty:
                os.makedirs(os.path.dirname(cached_csv), exist_ok=True)
                expr_df.to_csv(cached_csv, index=False)
                print(f"Cached results saved to {cached_csv}")

        if not expr_df.empty:
            table_tex = build_expressions_intervention_table(expr_df)
            tex_file = os.path.join(expr_interv_output, 'expressions_intervention.tex')
            with open(tex_file, 'w') as f:
                f.write(table_tex)
            print(f"Expressions intervention table saved to: {tex_file}")
        else:
            print("No expressions intervention results found.")
    except Exception as e:
        print(f"Error processing expressions intervention results: {e}")
        import traceback
        traceback.print_exc()


    #################################################
    ######### Extract Equation Examples #############
    #################################################

    print("\n" + "="*70)
    print("EXTRACTING EQUATION EXAMPLES")
    print("="*70)

    # Define paths to extract from
    equation_example_paths = [
        f"{output_path}/sr_ablation",
        f"{output_path}/prior_reg",
        f"{output_path}/memory_cls",
        f"{output_path}/memory_less_cls",
        f"{output_path}/memory_reg",
        f"{output_path}/memory_less_reg",
    ]

    # Define fixed class for classification datasets
    # Using class 0 for all classification datasets as default
    fixed_class_map = {
        'awa2': 0,
        'awa2_incomplete': 0,
        'cub': 0,
        'cub_incomplete': 0,
        'cifar10': 0,
    }

    try:
        from plot_utils import extract_equation_examples
        extract_equation_examples(
            paths=equation_example_paths,
            output_path=os.path.join(table_path, 'equation_examples'),
            n_examples=5,
            fixed_class=fixed_class_map,
            fixed_memory=fixed_memory
        )
        print("\n✓ Equation examples extracted successfully!")
    except Exception as e:
        print(f"Error occurred while extracting equation examples: {e}")
        import traceback
        traceback.print_exc()


    ##################################################
    ######### Explanation Robustness Tables ###########
    ##################################################

    print("\n" + "="*70)
    print("EXPLANATION ROBUSTNESS TABLES")
    print("="*70)

    robustness_base_path = f"{output_path}/global_interpretability"
    robustness_output = os.path.join(table_path, 'explanation_robustness')
    os.makedirs(robustness_output, exist_ok=True)

    try:
        rob_df = collect_explanation_robustness_results(robustness_base_path)
        if not rob_df.empty:
            print(f"Collected {len(rob_df)} rows from explanation robustness experiments.")
            print(f"Models: {rob_df['model'].unique()}")
            print(f"Datasets: {rob_df['dataset'].unique()}")
            print(f"Alphas: {sorted(rob_df['alpha'].unique())}")

            # Save aggregated raw results
            rob_df.to_csv(os.path.join(robustness_output, 'explanation_robustness_all.csv'), index=False)

            for agg in ['mean', 'max']:
                tables = build_explanation_robustness_tables(
                    rob_df, model_styles, agg=agg, custom_order=custom_order
                )
                for alpha, tex in tables.items():
                    fname = f"robustness_{agg}_alpha_{alpha:.2f}.tex"
                    tex_path = os.path.join(robustness_output, fname)
                    with open(tex_path, 'w') as f:
                        f.write(tex)
                print(f"Generated {len(tables)} {agg}-tables for explanation robustness.")

                # Normalized tables
                if 'lipschitz_constant_normalized' in rob_df.columns:
                    tables_norm = build_explanation_robustness_tables(
                        rob_df, model_styles, agg=agg, custom_order=custom_order,
                        metric_col='lipschitz_constant_normalized'
                    )
                    for alpha, tex in tables_norm.items():
                        fname = f"robustness_norm_{agg}_alpha_{alpha:.2f}.tex"
                        tex_path = os.path.join(robustness_output, fname)
                        with open(tex_path, 'w') as f:
                            f.write(tex)
                    print(f"Generated {len(tables_norm)} {agg}-tables for normalized explanation robustness.")

            print(f"Tables saved to {robustness_output}")
        else:
            print("No explanation robustness results found.")
    except Exception as e:
        print(f"Error occurred while processing explanation robustness results: {e}")
        import traceback
        traceback.print_exc()


    ####################################################################
    #### Intervention p_int Table (CUB200, etc.) for boolean models ####
    ####################################################################
    
    print("\n" + "="*70)
    print("INTERVENTION P_INT TABLE")
    print("="*70)
    
    paths = [
        f"{output_path}/memory_less_cls",
        f"{output_path}/memory_cls",
        f"{output_path}/boolean_m_cbe",
    ]
    
    try:
        # Load intervention data
        performance = get_intervention_from_path(paths)
        
        if not performance.empty:
            # Filter the performance dataframe to keep only the models in model_styles 
            # and datasets in custom_order
            performance = performance[performance['model'].isin(model_styles.keys()) & 
                                    performance['dataset'].isin(custom_order)]
            
            # Define models and datasets to include
            models_to_include = ['dcr', 'cmr', 'bool_symbolic_cbm']
            datasets_to_include = ['cub']
            memory_size_to_include = 1
            
            # Generate intervention p_int table
            intervention_pint_output = os.path.join(table_path, 'intervention_pint')
            os.makedirs(intervention_pint_output, exist_ok=True)
            
            pint_results = generate_intervention_pint_table(
                performance,
                model_styles,
                intervention_pint_output,
                models_to_include=models_to_include,
                datasets_to_include=datasets_to_include,
                noise_level=0.0,
                memory_size=memory_size_to_include
            )
            
            if pint_results is not None:
                print(f"\n✓ Intervention p_int table generated successfully!")
                print(f"\nSummary:")
                print(f"Models: {models_to_include}")
                print(f"Datasets: {datasets_to_include}")
            else:
                print("Failed to generate intervention p_int table")
        else:
            print("No intervention data found")
            
    except Exception as e:
        print(f"Error occurred while generating intervention p_int table: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()