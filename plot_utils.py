import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import torch
import pandas as pd
import scienceplots
import warnings
import yaml
from tqdm import tqdm
from show_results import table_path, result_figs, regression_datasets

# I used scienceplots for the style of the plots, but you can use any other style you want.
plt.style.use(['science', 'ieee', 'no-latex'])

########################################
########## Name conversion #############
########################################

def get_df_name(df):
    if df=='xor':
        return 'XOR'
    elif df=='dot':
        return 'DOT'
    elif df=='trigonometry':
        return 'Trigonometry'
    elif df=='and':
        return 'AND'
    elif df=='or':
        return 'OR'
    elif df=='cebab':
        return 'CEBaB'
    elif df=='mnist_addition':
        return 'MNIST+'
    elif df=='mnist_addition_incomplete':
        return 'MNIST-Add-Inc.'
    elif df=='cub':
        return 'CUB200'
    elif df=='imdb':
        return 'IMDB'
    elif df=='celeba':
        return 'CelebA'
    elif df=='mnist_even_odd':
        return 'MNIST-E/O'
    elif df=='awa2':
        return 'AWA2'
    elif df=='checkmark':
        return 'Checkmark'
    elif df=='celeba':
        return 'CelebA'
    elif df=='cub_incomplete':
        return 'CUB200-Incomplete'
    elif df=='awa2_incomplete':
        return 'AWA2-Incomplete'
    elif df=='cifar10':
        return 'CIFAR10'
    elif df=='cifar100':
        return 'CIFAR100'
    elif df=='dsprites_simple':
        return 'dSprites-Exp.'
    elif df=='dsprites_complex':
        return 'dSprites-Composed'
    elif df=='mnist_arithmetic':
        return 'MNIST-Arith.'
    elif df=='pendulum':
        return 'Pendulum'
    elif df=='mawps':
        return 'MAWPS'
    elif df=='feynman_I_6_2':
        return 'Feynman I.6.20'
    elif df=='feynman_I_9_18':
        return 'Feynman I.9.18'
    elif df=='feynman_I_12_1':
        return 'Feynman I.12.1'
    elif df=='feynman_I_13_4':
        return 'Feynman I.13.4'
    elif df=='feynman_I_14_3':
        return 'Feynman I.14.3'
    elif df=='feynman_I_15_10':
        return 'Feynman I.15.10'

classes_x_dataset = {
    'cub': 200,
    'cub_incomplete': 200,
    'awa2': 50,
    'awa2_incomplete': 50,
    'cifar10': 10,
    'cifar100': 100,
}


#########################################
######### Data Extraction ###############
#########################################

def get_exp_from_path_cached(paths, cache_name, output_path, sample_equations=False):
    """
    Cached wrapper for get_exp_from_path function.
    
    Args:
        paths: List of paths to process
        cache_name: Name identifier for this experiment (e.g., 'sr_ablation', 'memory_ablation')
        output_path: Base output path where cache directory will be created
        sample_equations: Whether to sample equations
    
    Returns:
        Tuple of (performance_df, other_data) as returned by get_exp_from_path
    """
    # Create cache directory
    cache_dir = os.path.join(output_path, 'cached_results')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filename based on experiment name and parameters
    cache_file = os.path.join(cache_dir, f'{cache_name}_equations_{sample_equations}.csv')
    
    # Check if cache exists
    if os.path.exists(cache_file):
        print(f"Loading cached results for {cache_name} from {cache_file}")
        performance = pd.read_csv(cache_file)
        # Return tuple to match original function signature
        return performance, None
    else:
        print(f"No cache found for {cache_name}. Computing results...")
        # Call the original function
        performance, other_data = get_exp_from_path(paths, sample_equations=sample_equations)
        
        # Save to cache
        print(f"Saving results to cache: {cache_file}")
        performance.to_csv(cache_file, index=False)
        
        return performance, other_data

def get_exp_from_path(paths, sample_equations=False):
    from src.utils.complexity import compute_complexity
    from sympy import sympify, symbols
    
    # Collect all the experiments in the given paths
    exps_path = []
    lmr_paths = []
    for path in paths:
        if os.path.exists(path):
            experiment_dir = os.listdir(path)
            for exp in experiment_dir:
                exps_path += [os.path.join(path, exp, e) for e in os.listdir(os.path.join(path, exp)) if 'multirun' not in e]

    performance = pd.DataFrame()

    # Iterate over all the experiments and collect the performance metrics and the config
    for exp in tqdm(exps_path, desc="Processing experiments"):
        d = {}
        conf_file = os.path.join(exp, '.hydra/config.yaml')
        result_file = os.path.join(exp, 'logs/experiment_metrics/metrics.csv') 
        try: 
            if os.path.exists(conf_file) and os.path.exists(result_file):
                with open(conf_file, 'r') as file:
                    conf = yaml.safe_load(file)
                d['seed'] = conf['seed']
                d['dataset'] = conf['dataset']['metadata']['name']
                d['model'] = conf['model']['metadata']['name']
                d['memory_size'] = conf['memory_size']
                d['concept_percentage'] = conf['concept_percentage']
                # get also the task type (classification or regression)
                d['task_type'] = conf['dataset']['metadata']['task']
                d['path'] = exp

                with open(result_file, 'r') as file:
                    result = pd.read_csv(file)

                # Select the last row of the dataframe where we test the model
                # if 'test/y/acc' and 'test_concept_acc' are not in the dataframe, skip the experiment
                if 'test/y/mse' in result.columns:
                    d['task_mse'] = result['test/y/mse'].iloc[-1]
                    d['task_mae'] = result['test/y/mae'].iloc[-1]
                else:
                    d['task_acc'] = result['test/y/acc'].iloc[-1]

                if conf['model']['metadata']['name']=='blackbox':
                    d['concept'] = 0
                else:
                    if 'test/c/mse' in result.columns:
                        d['concept_mse'] = result['test/c/mse'].iloc[-1]
                        d['concept_mae'] = result['test/c/mae'].iloc[-1]
                    else:
                        d['concept_acc'] = result['test/c/acc'].iloc[-1]
                
                # Compute complexity for equations if available (skip for cem and blackbox)
                if conf['model']['metadata']['name'] not in ['cem', 'blackbox']:
                    predictions_file = os.path.join(exp, 'logs/experiment_metrics/test_predictions_per_sample.csv')
                    if os.path.exists(predictions_file):
                        try:
                            df_pred = pd.read_csv(predictions_file)
                            
                            # Check if 'equation' column exists
                            if 'equation' in df_pred.columns:
                                # The variables of the equations are the concept names
                                vars = [x.replace('c_pred_','') for x in df_pred.columns if 'c_pred' in x]
                                
                                # Count unique equations and their occurrences
                                equation_counts = df_pred['equation'].value_counts().to_dict()

                                # Remove NaN or empty equations
                                equation_counts = {eq: cnt for eq, cnt in equation_counts.items() if pd.notna(eq) and eq != ''}

                                if sample_equations and len(equation_counts) >= 10:
                                    # Filter equation_counts to keep the top 10 most frequent equations
                                    equation_counts = dict(sorted(equation_counts.items(), key=lambda item: item[1], reverse=True)[:10])

                                # Remove target from equations if present (e.g., "y: <equation>")
                                cleaned_equation_counts = {}
                                for eq, cnt in equation_counts.items():
                                    if ':' in eq:
                                        cleaned_eq = eq.split(':')[1].strip()
                                    else:
                                        cleaned_eq = eq
                                    # Sum counts if multiple equations clean to the same form
                                    if cleaned_eq in cleaned_equation_counts:
                                        cleaned_equation_counts[cleaned_eq] += cnt
                                    else:
                                        cleaned_equation_counts[cleaned_eq] = cnt

                                # Compute complexity for each unique equation
                                total_complexity = 0
                                total_count = 0
                                for equation_str, count in tqdm(cleaned_equation_counts.items(), desc=f"Computing complexity ({d['dataset']}/{d['model']}/seed{d['seed']})", leave=False):
                                    try:
                                        # Convert string to sympy expression
                                        # Create proper symbols for all variables to avoid conflicts with built-in names
                                        symbol_dict = {v: symbols(v) for v in vars}
                                        equation = sympify(equation_str, locals=symbol_dict)
                                        
                                        # Compute complexity
                                        complexity = compute_complexity(equation, metric='visitation_length')
                                        
                                        # Weight by occurrence count
                                        total_complexity += complexity * count
                                        total_count += count
                                    except Exception as e:
                                        print(f"Error processing equation '{equation_str}': {e}")
                                        continue

                                if total_count > 0:
                                    # Compute weighted average complexity
                                    d['complexity'] = total_complexity / total_count
                                    d['n_equations'] = total_count
                        except Exception as e:
                            print(f"Error computing complexity for {exp}: {e}")
                else:
                    # Store NaN for blackbox and cem models
                    d['complexity'] = np.nan
                    d['n_equations'] = np.nan
                
                if d['model'] == 'linear_symbolic_cbm' and d['seed']==1:
                    expl_dict = d.copy()
                    expl_dict['path'] = exp
                    lmr_paths.append(expl_dict)

                performance = pd.concat([performance, pd.DataFrame([d])], ignore_index=True)
        except Exception as e:
            print(f"Error while processing {exp}: {e}")
            print(f"Skipping this experiment: {d}")
            continue

    return performance, lmr_paths

def get_intervention_from_path(paths, model_styles=None, custom_order=None, apply_filter=False, fixed_memory=None, selected_memory_size=2):

    memory_models = ['cmr', 'linear_symbolic_cbm', 'sr_symbolic_cbm', 'prior_symbolic_cbm', 'memory_cbm']

    if apply_filter:
        # Eliminate Feynman datasets from custom_order
        custom_order = [d for d in custom_order if not d.startswith('feynman')]

        # Create the filtered_exps dict
        # If the dataset in fixed memory, filter the experiments to keep only those with memory_size equal to the fixed memory
        filtered_exps = []
        if fixed_memory is not None:
            for dataset in custom_order:
                if dataset not in fixed_memory.keys():
                    for model in model_styles.keys():
                        if model in memory_models:
                            filtered_exps.append({
                                'dataset': dataset,
                                'model': model,
                                'memory_size': selected_memory_size
                            })
                else:
                    for model in model_styles.keys():
                        if model in memory_models:
                            mem_size = fixed_memory[dataset]
                            filtered_exps.append({
                                'dataset': dataset,
                                'model': model,
                                'memory_size': mem_size
                        }) 
        else:          
            for dataset in custom_order:
                for model in model_styles.keys():
                    if model in memory_models:
                        filtered_exps.append({
                        'dataset': dataset,
                        'model': model,
                        'memory_size': selected_memory_size
                    })

    performance = pd.DataFrame()

    # Collect all the experiments in the given paths
    exps_path = []
    lmr_paths = []
    
    for path in paths:
        if os.path.exists(path):
            experiment_dir = os.listdir(path)
            for exp in experiment_dir:
                exps_path += [os.path.join(path, exp, e) for e in os.listdir(os.path.join(path, exp)) if 'multirun' not in e]

    for exp in exps_path:
        conf_file = os.path.join(exp, '.hydra/config.yaml')
        result_file = os.path.join(exp, 'logs/experiment_metrics/interventions.csv')        
        if os.path.exists(conf_file) and os.path.exists(result_file):
            with open(result_file, 'r') as file:
                d = pd.read_csv(result_file)[['noise','p_int','f1','accuracy','mae','mse']]
            
            with open(conf_file, 'r') as file:
                conf = yaml.safe_load(file)
            d['seed'] = conf['seed']
            d['dataset'] = conf['dataset']['metadata']['name']
            d['model'] = conf['model']['metadata']['name']
            d['memory_size'] = conf['memory_size']

            performance = pd.concat([performance, d], ignore_index=True)

    if apply_filter is False:
        return performance
    else:
        # Keep only the experiments in filtered_exps
        criteria = pd.DataFrame(filtered_exps)
        filtered_performance = performance.merge(criteria, on=criteria.columns.tolist(), how="inner")

        # If the model belong to cmb_linear, licem, dcr, cem, blackbox, add them to performance
        memoryless_models = performance[performance['model'].isin(['cmb_linear', 'cbm_mlp', 'licem', 'dcr', 'cem', 'blackbox'])]
        filtered_performance = pd.concat([filtered_performance, memoryless_models], ignore_index=True)
        return filtered_performance

##########################################
######## Data Processing #################
##########################################

def compute_avg_and_uncertainty(performance, custom_order):

    # Sort the points in the order of memory_size.
    performance = performance.sort_values(by=['dataset', 'memory_size'])

    # Set cmb_linear's memory_size to 1 and licem/dcr's memory_size to 500
    performance.loc[performance['model'].isin(['cmb_linear', 'cem', 'blackbox']), 'memory_size'] = 1
    performance.loc[performance['model'].isin(['licem', 'dcr']), 'memory_size'] = 500

    num_seeds = performance['seed'].nunique()

    # Multiply the accuracy values by 100 to convert them to percentages
    if 'task_acc' in performance.columns:
        performance['task_acc'] = performance['task_acc'] * 100

    # Avg over the seeds for the performance metrics
    if 'task_acc' in performance.columns and 'task_mae' in performance.columns:
        # Handle both accuracy and MSE/MAE metrics
        agg_dict_acc = {
            'mean_task': ('task_acc', 'mean'),
            'std_task': ('task_acc', 'std'),
            'mean_concept': ('concept_acc', 'mean'),
            'std_concept': ('concept_acc', 'std'),
            'task_type': ('task_type', 'first')
        }
        if 'complexity' in performance.columns:
            agg_dict_acc['mean_complexity'] = ('complexity', 'mean')
            agg_dict_acc['std_complexity'] = ('complexity', 'std')
        
        performance_acc = performance.dropna(subset=['task_acc']).groupby(['dataset', 'memory_size', 'model']).agg(
            **agg_dict_acc
        ).reset_index()
        performance_acc['metric_type'] = 'accuracy'
        
        agg_dict_mae = {
            'mean_task': ('task_mae', 'mean'),
            'std_task': ('task_mae', 'std'),
            'mean_concept': ('concept_mae', 'mean'),
            'std_concept': ('concept_mae', 'std'),
            'task_type': ('task_type', 'first')
        }
        if 'complexity' in performance.columns:
            agg_dict_mae['mean_complexity'] = ('complexity', 'mean')
            agg_dict_mae['std_complexity'] = ('complexity', 'std')
        
        performance_mae = performance.dropna(subset=['task_mae']).groupby(['dataset', 'memory_size', 'model']).agg(
            **agg_dict_mae
        ).reset_index()
        performance_mae['metric_type'] = 'mae'
        
        performance = pd.concat([performance_acc, performance_mae], ignore_index=True)
    else:
        # Fallback to original logic
        task_col = 'task_acc' if 'task_acc' in performance.columns else 'task_mae'
        concept_col = 'concept_acc' if 'concept_acc' in performance.columns else 'concept_mae'
        agg_dict_fallback = {
            'mean_task': (task_col, 'mean'),
            'std_task': (task_col, 'std'),
            'mean_concept': (concept_col, 'mean'),
            'std_concept': (concept_col, 'std'),
            'task_type': ('task_type', 'first')
        }
        if 'complexity' in performance.columns:
            agg_dict_fallback['mean_complexity'] = ('complexity', 'mean')
            agg_dict_fallback['std_complexity'] = ('complexity', 'std')
        
        performance = performance.groupby(['dataset', 'memory_size', 'model']).agg(
            **agg_dict_fallback
        ).reset_index()
        performance['metric_type'] = 'accuracy' if 'acc' in task_col else 'mae'

    # instead of the std compute the standard error at 95% confidence
    performance['se_task'] = 1.96 * performance['std_task'] / np.sqrt(num_seeds)
    performance['se_concept'] = 1.96 * performance['std_concept'] / np.sqrt(num_seeds)
    
    # Compute standard error for complexity if it exists
    if 'std_complexity' in performance.columns:
        performance['se_complexity'] = 1.96 * performance['std_complexity'] / np.sqrt(num_seeds)

    # Order the datasets according to custom_order and memory_size
    performance['dataset'] = pd.Categorical(performance['dataset'], categories=custom_order, ordered=True)
    performance = performance.sort_values(['dataset', 'memory_size'])

    return performance

###########################################
######### Plotting Interventions ##########
###########################################

def plot_intervention_results(
    df, 
    metric='accuracy', 
    unique_noises=[0.0], 
    title_font=None, 
    label_font=None, 
    tick_font=None, 
    legend_font=None,
    custom_order=None,
    model_styles=None,
    relative_accuracy=True,
    out_dir=None,
):

    unique_datasets = [d for d in df['dataset'].unique()]

    # # sort the organized_datasets according to custom_order
    # organized_datasets = [d for d in custom_order if d in organized_datasets]

    # Separate datasets by task type
    classification_datasets = [d for d in unique_datasets if d not in regression_datasets]
    found_regression_datasets = [d for d in unique_datasets if d in regression_datasets]

    # Orgsanize dataset based on custom order
    classification_datasets = [d for d in custom_order if d in classification_datasets]
    found_regression_datasets = [d for d in custom_order if d in found_regression_datasets]
    
    # Determine if we have both types of datasets
    has_classification = len(classification_datasets) > 0
    has_regression = len(found_regression_datasets) > 0
    
    # If only one type exists, use both rows for that type
    if has_classification and not has_regression:
        # Only classification datasets - spread across 2 rows
        organized_datasets = classification_datasets
        n_cols = (len(classification_datasets) + 1) // 2  # Ceiling division
        n_rows = 2
    elif has_regression and not has_classification:
        # Only regression datasets - spread across 2 rows
        organized_datasets = found_regression_datasets
        n_cols = (len(found_regression_datasets) + 1) // 2  # Ceiling division
        n_rows = 2
    else:
        # Both types exist - organize with regression first, then classification
        organized_datasets = found_regression_datasets + classification_datasets
        n_cols = max(len(classification_datasets), len(found_regression_datasets))
        n_rows = 2  # Force 2 rows: regression on first row, classification on second
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows), sharex=False, sharey=False)

    # Ensure axes is always a 2D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])
    else:
        axes = np.array(axes)
    
    for idx, dataset in enumerate(organized_datasets):
        # Determine row and column based on layout
        if has_classification and not has_regression:
            # Only classification - spread across 2 rows
            row = idx // n_cols
            col = idx % n_cols
        elif has_regression and not has_classification:
            # Only regression - spread across 2 rows
            row = idx // n_cols
            col = idx % n_cols
        else:
            # Both types - regression on row 0, classification on row 1
            if dataset in found_regression_datasets:
                row = 0
                col = found_regression_datasets.index(dataset)
            else:
                row = 1
                col = classification_datasets.index(dataset)
        
        # Access the axis consistently
        ax = axes[row][col]
        
        for noise in unique_noises:
            data = df[(df['noise'] == noise) & (df['dataset'] == dataset)]
            if dataset in found_regression_datasets:
                metric = 'mae'
            else:
                metric = 'accuracy'
            grouped_data = data.groupby(['p_int', 'model']).agg(
                mean_metric=(metric, 'mean'),
                std_metric=(metric, 'std')
            ).reset_index().fillna(0)
            
            # Calculate number of seeds for standard error
            num_seeds = data.groupby(['p_int', 'model']).size().reset_index(name='n_seeds')
            grouped_data = grouped_data.merge(num_seeds, on=['p_int', 'model'])
            grouped_data['se_metric'] = 1.96 * grouped_data['std_metric'] / np.sqrt(grouped_data['n_seeds'])
            
            # Convert accuracy to error rate for classification tasks (accuracy is in decimal form 0-1)
            if dataset not in found_regression_datasets:
                grouped_data['mean_metric'] = 100 * (1 - grouped_data['mean_metric'])
            
            if relative_accuracy:
                # Calculate relative accuracy for each model
                for model in grouped_data['model'].unique():
                    model_data = grouped_data[grouped_data['model'] == model]
                    baseline = model_data[model_data['p_int'] == 0]['mean_metric'].iloc[0] if len(model_data[model_data['p_int'] == 0]) > 0 else 0
                    grouped_data.loc[grouped_data['model'] == model, 'mean_metric'] -= baseline
            
            for model in grouped_data['model'].unique():
                model_data = grouped_data[grouped_data['model'] == model]
                style = model_styles.get(model, {'marker': 'o', 'color': 'black', 'size': 10, 'name': model})
                
                # Plot the line and markers
                ax.plot(model_data['p_int'], model_data['mean_metric'],
                       marker=style['marker'], 
                       color=style['color'], 
                       markersize=style['size'], 
                       label=style['name'],
                       markeredgecolor='black',
                       markeredgewidth=0.1,
                       alpha=0.8,
                       linestyle='-')
                
                # Add shaded area for uncertainty
                ax.fill_between(model_data['p_int'],
                               model_data['mean_metric'] - model_data['se_metric'],
                               model_data['mean_metric'] + model_data['se_metric'],
                               color=style['color'],
                               alpha=0.2)
        
        # Set x-axis ticks and labels
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
        
        # Show xlabel only for the last row
        if has_classification and not has_regression:
            # Only classification - show xlabel on last row
            if row == n_rows - 1 or idx >= len(organized_datasets) - n_cols:
                ax.set_xlabel('$p_{int}$', fontsize=label_font['size'])
        elif has_regression and not has_classification:
            # Only regression - show xlabel on last row
            if row == n_rows - 1 or idx >= len(organized_datasets) - n_cols:
                ax.set_xlabel('$p_{int}$', fontsize=label_font['size'])
        else:
            # Both types - show xlabel on row 1 (classification row)
            if row == 1:
                ax.set_xlabel('$p_{int}$', fontsize=label_font['size'])
        
        ax.set_title(f'{get_df_name(dataset)}', fontsize=label_font['size'])

        # Set ylabel only for the leftmost subplot in each row
        if col == 0:
            if has_classification and not has_regression:
                # Only classification datasets
                ylabel = '$\Delta$ (Error Rate)' if relative_accuracy else 'Error Rate'
                ax.set_ylabel(ylabel, fontsize=label_font['size'])
            elif has_regression and not has_classification:
                # Only regression datasets
                ylabel = '$\Delta$ MAE' if relative_accuracy else 'MAE'
                ax.set_ylabel(ylabel, fontsize=label_font['size'])
            else:
                # Both types - set ylabel based on dataset type
                if dataset in found_regression_datasets:
                    ylabel = '$\Delta$ MAE' if relative_accuracy else 'MAE'
                    ax.set_ylabel(ylabel, fontsize=label_font['size'])
                else:
                    ylabel = '$\Delta$ (Error Rate)' if relative_accuracy else 'Error Rate'
                    ax.set_ylabel(ylabel, fontsize=label_font['size'])
        
        ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
        ax.minorticks_off()
        ax.grid(True, alpha=0.3, zorder=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # Hide empty subplots and check if we can place legend in missing subplot
    legend_in_subplot = False
    legend_ax = None
    
    if has_classification and not has_regression:
        # Only classification - hide unused subplots in both rows
        total_used = len(organized_datasets)
        for i in range(total_used, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row][col].set_visible(False)
            # Use the last empty subplot for legend
            if i == n_rows * n_cols - 1 and total_used < n_rows * n_cols:
                legend_in_subplot = True
                legend_ax = axes[row][col]
    elif has_regression and not has_classification:
        # Only regression - hide unused subplots in both rows
        total_used = len(organized_datasets)
        for i in range(total_used, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row][col].set_visible(False)
            # Use the last empty subplot for legend
            if i == n_rows * n_cols - 1 and total_used < n_rows * n_cols:
                legend_in_subplot = True
                legend_ax = axes[row][col]
    else:
        # Both types exist - original logic
        total_datasets = len(classification_datasets) + len(found_regression_datasets)
        
        # Hide unused subplots in first row (regression)
        for col in range(len(found_regression_datasets), n_cols):
            axes[0][col].set_visible(False)
            # Use the rightmost empty subplot in first row for legend if available
            if col == n_cols - 1:
                legend_in_subplot = True
                legend_ax = axes[0][col]
        
        # Hide unused subplots in second row (classification) and check for legend placement
        for col in range(len(classification_datasets), n_cols):
            axes[1][col].set_visible(False)
            # Use the rightmost empty subplot in second row for legend if first row doesn't have one
            if legend_ax is None and col == n_cols - 1:
                legend_in_subplot = True
                legend_ax = axes[1][col]
    
    # Filter the style according to the models' names which are present in the df
    filtered_styles = {name: style for name, style in model_styles.items() if name in df['model'].values}

    # Create custom legend handles
    custom_handles = [plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=style['size']+10, label=style['name'], markeredgewidth=0.5, markeredgecolor='black') for style in filtered_styles.values()]

    # Place legend either in empty subplot or below plots
    if legend_in_subplot:
        legend_ax.set_visible(True)
        legend_ax.axis('off')  # Hide axes
        legend_ax.legend(handles=custom_handles, loc='center', ncol=1, fontsize=tick_font['size'], frameon=True)
    else:
        # Create a single legend below the plots
        fig.legend(
            handles=custom_handles,
            loc='lower center',
            ncol=(len(custom_handles) + 1) // 2,  # Split legend into two rows
            fontsize=tick_font['size'],
            frameon=True,
            bbox_to_anchor=(0.5, -0.15),
            columnspacing=1.0,
            handletextpad=0.5
        )

    plt.tight_layout()
    str_store = str(unique_noises[0]).replace('.', '')
    suffix = 'relative_accuracy_difference' if relative_accuracy else 'absolute_accuracy'
    os.makedirs(f'{out_dir}/intervention/{suffix}', exist_ok=True)
    plt.savefig(f'{out_dir}/intervention/{suffix}/{str_store}.pdf')
    plt.show()

###########################################
######### Plotting Memory Ablation ########
###########################################

def plot_memory_ablation(performance, model_styles, title_font, label_font, tick_font, custom_order):

    performance = compute_avg_and_uncertainty(performance, custom_order)

    # Separate datasets by task type
    classification_datasets = performance[performance['task_type'] == 'classification']['dataset'].unique()
    found_regression_datasets = performance[performance['task_type'] == 'regression']['dataset'].unique()
    
    # Organize datasets with MAE first, then 1-accuracy
    organized_datasets = list(found_regression_datasets) + list(classification_datasets)
    
    n_datasets = len(organized_datasets)
    n_cols = max(len(classification_datasets), len(found_regression_datasets))  
    n_rows = 2  # Force 2 rows: MAE on first row, 1-accuracy on second

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows), sharey=False)
    
    # Handle case where we have only one subplot
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for idx, dataset in enumerate(organized_datasets):
        # Determine row based on task type
        if dataset in found_regression_datasets:
            row = 0
            col = list(found_regression_datasets).index(dataset)
        else:
            row = 1
            col = list(classification_datasets).index(dataset)
        
        if n_rows == 1:
            ax = axes[col] if n_cols > 1 else axes[0]
        else:
            ax = axes[row][col] if n_cols > 1 else axes[row][0] if col == 0 else axes[row][col]
            
        data = performance[performance['dataset'] == dataset]
        metric_type = data['metric_type'].iloc[0]
        
        # Get distinct memory sizes for this dataset
        distinct_memory_sizes = sorted(data['memory_size'].unique())
        
        for model in data['model'].unique():
            model_data = data[data['model'] == model]
            
            # Check if this model has unique memory size (only one data point)
            has_unique_memory_size = len(model_data) == 1
            
            # For accuracy metrics, plot 1-accuracy (error rate)
            if metric_type == 'accuracy':
                y_values = 100 - model_data['mean_task']
                y_errors = model_data['se_task']  # Error bars remain the same
            else:
                y_values = model_data['mean_task']
                y_errors = model_data['se_task']
            
            if has_unique_memory_size:
                # Show uncertainty as vertical error bars for single points
                ax.errorbar(
                    model_data['memory_size'], 
                    y_values,
                    yerr=y_errors,
                    label=model_styles[model]['name'], 
                    marker=model_styles[model]['marker'], 
                    color=model_styles[model]['color'], 
                    markersize=model_styles[model]['size'],
                    markeredgecolor='black',
                    markeredgewidth=0.1,
                    capsize=3,
                    capthick=1.5,
                    alpha=0.8,
                    linestyle='none'  # No line for single points
                )
            else:
                # Show uncertainty as shaded area for multiple points
                ax.plot(
                    model_data['memory_size'], 
                    y_values,
                    label=model_styles[model]['name'], 
                    marker=model_styles[model]['marker'], 
                    color=model_styles[model]['color'], 
                    markersize=model_styles[model]['size'],
                    markeredgecolor='black',
                    markeredgewidth=0.1,
                    alpha=0.8
                )
                ax.fill_between(
                    model_data['memory_size'],
                    y_values - y_errors,
                    y_values + y_errors,
                    color=model_styles[model]['color'],
                    alpha=0.2
                )
        
        ax.set_title(get_df_name(dataset), fontdict=title_font)

        if row==1:
            ax.set_xlabel('Memory Size', fontdict=label_font)
        else:
            ax.set_xlabel('')
        
        # Set y-label only for the leftmost subplot in each row
        if col == 0:
            ylabel = 'MAE' if metric_type == 'mae' else 'Error Rate'
            ax.set_ylabel(ylabel, fontdict=label_font)
        
        # Set x-axis to log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Set x-axis ticks and labels with infinity symbol for memory_size=500
        ax.set_xticks(distinct_memory_sizes)
        x_labels = []
        for size in distinct_memory_sizes:
            if size == 500:
                x_labels.append('$\dots\infty$')
            else:
                x_labels.append(str(int(size)))
        ax.set_xticklabels(x_labels)
        
        ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
        ax.minorticks_off()
        ax.grid(True, alpha=0.3, zorder=0)

    # Hide empty subplots and find the rightmost empty subplot for legend
    legend_ax = None
    
    # Hide unused subplots in first row (MAE datasets)
    for col in range(len(found_regression_datasets), n_cols):
        axes[0][col].set_visible(False)
        # Use the rightmost empty subplot in first row for legend if available
        if col == n_cols - 1:
            legend_ax = axes[0][col]
    
    # Hide unused subplots in second row (1-accuracy datasets)
    for col in range(len(classification_datasets), n_cols):
        axes[1][col].set_visible(False)
        # Use the rightmost empty subplot in second row for legend if first row doesn't have one
        if legend_ax is None and col == n_cols - 1:
            legend_ax = axes[1][col]

    # Filter the style according to the models' names which are present in the performance df
    filtered_styles = {name: style for name, style in model_styles.items() if name in performance['model'].values}

    # Create custom legend handles
    custom_handles = [plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=(style['size'])+10, label=style['name'], markeredgewidth=0.5, markeredgecolor='black') for style in filtered_styles.values()]

    # Place legend in empty subplot if available, otherwise below plots
    if legend_ax is not None:
        legend_ax.set_visible(True)
        legend_ax.axis('off')  # Hide axes
        legend_ax.legend(handles=custom_handles, loc='center', ncol=1, fontsize=tick_font['size'], frameon=True)
    else:
        # Create a single legend below the plots with one model per column
        fig.legend(handles=custom_handles, loc='lower center', ncol=len(custom_handles), fontsize=tick_font['size'], frameon=True, bbox_to_anchor=(0.5, -0.15), columnspacing=1.0, handletextpad=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(result_figs, 'memory_ablation.pdf'))

    plt.show()


######################################
####### Concept Size Ablation ########
######################################

def plot_concept_size_ablation(
        performance, 
        model_styles, 
        title_font, 
        label_font, 
        tick_font):
    """
    Plots the concept size ablation results.
    In this plot the concept accuracy is shown on the y axis while the
    concept size (percentage) is shown on the x axis.
    Each line represent a different model and each subplot the results on a different dataset
    """

    # Sort the points in the order of concept_percentage.
    performance = performance.sort_values(by=['dataset', 'concept_percentage'])

    num_seeds = performance['seed'].nunique()

    # Avg over the seeds for the performance metrics
    performance = performance.groupby(['dataset', 'concept_percentage', 'model']).agg(
        mean_task=('task', 'mean'),
        std_task=('task', 'std'),
        mean_concept=('concept', 'mean'),
        std_concept=('concept', 'std')
    ).reset_index()
    
    # instead of the std compute the standard error at 95% confidence
    performance['se_task'] = 1.96 * performance['std_task'] / np.sqrt(num_seeds)
    performance['se_concept'] = 1.96 * performance['std_concept'] / np.sqrt(num_seeds)

    # Create a new figure
    fig, ax = plt.subplots(figsize=(25, 10))

    # Iterate over each model and plot its performance
    for model, style in model_styles.items():
        if model in performance['model'].values:
            data = performance[performance['model'] == model]
            ax.plot(data['concept_percentage'], data['mean_task'],
                    label=style['name'], marker=style['marker'], color=style['color'])
            ax.fill_between(data['concept_percentage'], 
                          data['mean_task'] - data['se_task'], 
                          data['mean_task'] + data['se_task'], 
                          color=style['color'], alpha=0.2)

    # Set the title and labels
    ax.set_title("Concept Size Ablation", **title_font)
    ax.set_xlabel("Dataset", **label_font)
    ax.set_ylabel("Accuracy", **label_font)

    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
    ax.grid(True)

    # Create legend below the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=len([m for m in model_styles.keys() if m in performance['model'].values]),
              fontsize=tick_font['size'], frameon=True)

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(result_figs, 'concept_size_ablation.pdf'), bbox_inches='tight')



############################################
############## Pareto plots ################
############################################

def filter_pareto_models(df: pd.DataFrame, fixed_memory: dict = None, custom_order: list = None, delta_threshold: float = 0.1) -> pd.DataFrame:
    """
    Filters a dataframe of model results to keep only specific memory_size per model/dataset/seed:
    
    For classification datasets:
        - dcr, licem, cem, blackbox: take all experiments (all memory sizes)
        - Other memory-based models: take only memory_size = 2
    
    For regression datasets:
        - licem, cem, blackbox: take all experiments (all memory sizes)
        - Other memory-based models: use memory size from fixed_memory dictionary
    
    If `fixed_memory` is provided (dict mapping dataset -> memory_size), it's used for
    regression datasets with memory-based models.
    """

    df = compute_avg_and_uncertainty(df, custom_order)

    fixed_memory = fixed_memory or {}
    
    # Models that should take all experiments
    classification_all_models = ['cbm_linear', 'cbm_mlp', 'dcr', 'licem', 'cem', 'blackbox']
    regression_all_models = ['cbm_linear', 'cbm_mlp', 'licem', 'cem', 'blackbox']

    def filter_by_rules(group: pd.DataFrame) -> pd.DataFrame:
        dataset = group["dataset"].iloc[0]
        model = group["model"].iloc[0]
        task_type = group["task_type"].iloc[0]

        if task_type == "classification":
            # For classification datasets
            if model in classification_all_models:
                # Take all experiments for dcr, licem, cem, blackbox
                return group
            else:
                # For other memory-based models, take only memory_size = 2
                filtered_group = group[group["memory_size"] == 2]
                if not filtered_group.empty:
                    return filtered_group
                else:
                    # If memory_size = 2 doesn't exist, return the group as is
                    return group
        
        elif task_type == "regression":
            # For regression datasets
            if model in regression_all_models:
                # Take all experiments for licem, cem, blackbox
                return group
            else:
                # For other memory-based models, use fixed_memory
                if dataset in fixed_memory:
                    mem_size = fixed_memory[dataset]
                    filtered_group = group[group["memory_size"] == mem_size]
                    if not filtered_group.empty:
                        return filtered_group
                    else:
                        # If specified memory size doesn't exist, return the group as is
                        return group
                else:
                    # If no fixed_memory specified, return all
                    return group
        else:
            return group

    # Apply per dataset-model
    filtered = df.groupby(["dataset", "model"], group_keys=False).apply(filter_by_rules)

    # Return a dataframe containing:
    # - dataset
    # - model
    # - memory_size
    filtered = filtered[["dataset", "model", "memory_size"]]

    return filtered.reset_index(drop=True)

def plot_pareto_front(performance, model_styles, title_font, label_font, tick_font, custom_order, complexity_type=False):
    """
    Plot Pareto front for model complexity vs accuracy.
    
    Parameters:
    -----------
    use_complexity_column : bool, default=False
        If True, use the 'complexity' column directly from the dataframe.
        If False, compute complexity as memory_size * operational_complexity.
    """
    performance = compute_avg_and_uncertainty(performance, custom_order)

    # save the performance as csv
    performance.to_csv(os.path.join(table_path, 'memory_ablation_performance.csv'), index=False)

    # Filter out dcr and cmr for cub200, awa2, and cifar10 datasets
    # performance = performance[~((performance['dataset'].isin(['cub', 'awa2', 'cifar10'])) & (performance['model'].isin(['dcr', 'cmr'])))]

    # Separate datasets by task type
    classification_datasets = performance[performance['task_type'] == 'classification']['dataset'].unique()
    found_regression_datasets = performance[performance['task_type'] == 'regression']['dataset'].unique()

    # Organize datasets with MAE first, then 1-accuracy
    organized_datasets = list(found_regression_datasets) + list(classification_datasets)
    
    n_datasets = len(organized_datasets)
    n_cols = max(len(classification_datasets), len(found_regression_datasets))  
    n_rows = 2  # Force 2 rows: MAE on first row, 1-accuracy on second

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows), sharey=False)
    
    # Handle case where we have only one subplot
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # Track which model styles are actually plotted across all subplots
    all_plotted_styles = set()

    for idx, dataset in enumerate(organized_datasets):
        # Determine row based on task type
        if dataset in found_regression_datasets:
            row = 0
            col = list(found_regression_datasets).index(dataset)
        else:
            row = 1
            col = list(classification_datasets).index(dataset)
        
        if n_rows == 1:
            ax = axes[col] if n_cols > 1 else axes[0]
        else:
            ax = axes[row][col] if n_cols > 1 else axes[row][0] if col == 0 else axes[row][col]
            
        data = performance[performance['dataset'] == dataset]
        metric_type = data['metric_type'].iloc[0]
        
        # Debug: print models present for this dataset
        print(f"Dataset: {dataset}, Models: {data['model'].unique()}")
        
        # Compute complexity for each model
        data = data.copy()
        data['has_infinity'] = False  # Initialize the column
        
        if complexity_type == 'oc' and 'mean_complexity' in data.columns:
            # Use the 'complexity' column directly from the dataframe
            # Mark dcr and licem as having infinity complexity
            data['has_infinity'] = data['model'].isin(['dcr', 'licem'])
            # Find max finite complexity to set infinity value appropriately
            finite_complexities = data.loc[~data['has_infinity'], 'mean_complexity']
            finite_complexities = finite_complexities[finite_complexities.notna() & (finite_complexities > 0)]
            
            if len(finite_complexities) > 0:
                max_finite = finite_complexities.max()
                infinity_value = max_finite * 100  # Use 100x the max as "infinity"
            else:
                # Fallback if no valid finite complexities exist
                infinity_value = 1000.0
            
            data.loc[data['has_infinity'], 'mean_complexity'] = infinity_value
        elif complexity_type == 'composed' and 'mean_complexity' in data.columns:
            # Compute complexity as memory_size * operational_complexity from dataframe column
            data['mean_complexity'] = data['memory_size'] * data['mean_complexity']
            # Mark dcr and licem as having infinity complexity, then replace with large finite value
            data['has_infinity'] = data['model'].isin(['dcr', 'licem'])
            # Find max finite complexity to set infinity value appropriately
            # Filter out NaN and negative values when finding max
            finite_complexities = data.loc[~data['has_infinity'], 'mean_complexity']
            finite_complexities = finite_complexities[finite_complexities.notna() & (finite_complexities > 0)]
            
            if len(finite_complexities) > 0:
                max_finite = finite_complexities.max()
                infinity_value = max_finite * 100  # Use 100x the max as "infinity"
            else:
                # Fallback if no valid finite complexities exist
                infinity_value = 1000.0
            
            data.loc[data['has_infinity'], 'mean_complexity'] = infinity_value
        else:
            raise ValueError("Either complexity_type is not supported or required columns are missing in the dataframe.")

        # Filter out rows with NaN or non-positive complexity values
        # BUT keep blackbox and cem models even if they have NaN complexity (they'll be plotted as horizontal lines)
        data = data[(data['mean_complexity'].notna() & (data['mean_complexity'] > 0)) | 
                    (data['model'].isin(['cem', 'blackbox']))]
        
        # Debug: check if data is empty after filtering
        if data.empty:
            print(f"WARNING: Dataset {dataset} has no valid complexity data after filtering!")
            continue
        
        # Get distinct complexities for this dataset (excluding cem and blackbox)
        pareto_data = data[~data['model'].isin(['cem', 'blackbox'])]
        distinct_complexities = sorted(pareto_data['mean_complexity'].unique())
        
        # Check if we have infinity models (dcr, licem) in the data
        has_infinity_models = any(data['model'].isin(['dcr', 'licem']))
        if has_infinity_models:
            infinity_complexity = data.loc[data['model'].isin(['dcr', 'licem']), 'mean_complexity'].unique()
        else:
            infinity_complexity = np.array([])
        
        # Collect all points for Pareto front calculation
        all_points = []
        
        for model in data['model'].unique():
            model_data = data[data['model'] == model]
            
            # For accuracy metrics, plot 1-accuracy (error rate)
            if metric_type == 'accuracy':
                y_values = 100 - model_data['mean_task']
                y_errors = model_data['se_task']  # Error bars remain the same
            else:
                y_values = model_data['mean_task']
                y_errors = model_data['se_task']
            
            if model in ['cem', 'blackbox']:
                # Plot as horizontal dotted lines for cem and blackbox
                xlim = ax.get_xlim() if ax.get_xlim() != (0.0, 1.0) else (1, 100000)  # Default range if not set
                y_val = y_values.iloc[0]
                
                ax.axhline(y=y_val, color=model_styles[model]['color'], 
                            linestyle='-.', linewidth=2.5, alpha=0.8,
                            label=model_styles[model]['name'])
                
                # add uncertainty shading for blackbox and cem
                y_err = y_errors.iloc[0]
                ax.fill_between(xlim,
                                y_val - y_err,
                                y_val + y_err,
                                color=model_styles[model]['color'],
                                alpha=0.2)
                
                # Track that this style was plotted
                all_plotted_styles.add(model)
                
            else:
                # Store points for Pareto front (excluding cem and blackbox)
                for i, (complexity, y_val, y_err, memory_size) in enumerate(zip(model_data['mean_complexity'], y_values, y_errors, model_data['memory_size'])):
                    all_points.append((complexity, y_val, y_err, model, memory_size))
                
                # Plot each point with appropriate style
                for i, (complexity, y_val, y_err, memory_size) in enumerate(zip(model_data['mean_complexity'], y_values, y_errors, model_data['memory_size'])):
                    # Use cbm_linear style if linear_symbolic_cbm has memory_size = 1
                    if model == 'linear_symbolic_cbm' and memory_size == 1:
                        plot_style = model_styles.get('cbm_linear', model_styles[model])
                        plot_label = model_styles.get('cbm_linear', {})['name'] if 'cbm_linear' in model_styles else model_styles[model]['name']
                        style_key = 'cbm_linear'
                    else:
                        plot_style = model_styles[model]
                        plot_label = model_styles[model]['name']
                        style_key = model
                    
                    # Track that this style was plotted
                    all_plotted_styles.add(style_key)
                    
                    # Only add label for the first point of each unique style to avoid duplicate legends
                    label = plot_label if i == 0 else ""
                    
                    # Show uncertainty as vertical error bars
                    ax.errorbar(
                        [complexity], 
                        [y_val],
                        yerr=[y_err],
                        label=label, 
                        marker=plot_style['marker'], 
                        color=plot_style['color'], 
                        markersize=plot_style['size'],
                        markeredgecolor='black',
                        markeredgewidth=0.1,
                        capsize=3,
                        capthick=1.5,
                        alpha=0.8,
                        linestyle='none'  # No line connecting dots
                    )
        
        # Calculate and draw Pareto front (excluding cem and blackbox)
        if all_points:
            # Sort points by complexity
            all_points.sort(key=lambda x: x[0])
            
            # Find Pareto optimal points
            pareto_points = []
            for point in all_points:
                complexity, y_val, y_err, model, memory_size = point
                is_pareto = True
                
                # For each point, check if it's dominated by any other point
                for other_point in all_points:
                    other_complexity, other_y_val, other_y_err, other_model, other_memory_size = other_point
                    
                    # A point is dominated if another point has both:
                    # - Lower or equal complexity AND lower performance metric (better)
                    if (other_complexity <= complexity and 
                        other_y_val < y_val and 
                        (other_complexity < complexity or other_y_val < y_val)):
                        is_pareto = False
                        break
                
                if is_pareto:
                    pareto_points.append((complexity, y_val))
            
            # Sort Pareto points by complexity and draw connecting line
            if len(pareto_points) > 1:
                pareto_points.sort(key=lambda x: x[0])
                pareto_x, pareto_y = zip(*pareto_points)
                ax.plot(pareto_x, pareto_y, 'gray', linestyle='--', alpha=0.7, linewidth=2, zorder=0)
                
                # Create shadowed area above and to the right of the Pareto front
                # Get axis limits to extend the shaded area
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                                
                # Create extended pareto front for shading
                extended_x = [pareto_x[0]] + list(pareto_x) + [xlim[1], xlim[1]]
                extended_y = [ylim[1]]     + list(pareto_y) + [pareto_y[-1], ylim[1]]

                # Add shaded area above the Pareto front
                ax.fill(extended_x, extended_y, color='gray', alpha=0.1, zorder=0, 
                        label='Dominated Region' if idx == 0 else "")
            
        ax.set_title(get_df_name(dataset), fontdict=title_font)

        if row==1:
            ax.set_xlabel('Complexity', fontdict=label_font)
        
        # Set y-label only for the leftmost subplot in each row
        if col == 0:
            ylabel = 'MAE' if metric_type == 'mae' else 'Error Rate'
            ax.set_ylabel(ylabel, fontdict=label_font)
        
        # Set x-axis to log scale
        ax.set_xscale('log')

        # Set x-axis ticks to 5 equally spaced values
        if distinct_complexities:
            # Separate finite and infinity complexities
            if has_infinity_models and infinity_complexity.size > 0:
                inf_value = infinity_complexity[0]
                finite_complexities = [c for c in distinct_complexities if c < inf_value]
            else:
                finite_complexities = distinct_complexities
            
            if finite_complexities:
                min_c = min(finite_complexities)
                max_c = max(finite_complexities)
                
                # Generate 5 equally spaced ticks in log space
                if has_infinity_models and infinity_complexity.size > 0:
                    # Calculate equal log spacing across all 5 ticks including infinity
                    # Infinity tick should be at the actual inf_value position
                    log_min = np.log10(min_c)
                    log_inf = np.log10(inf_value)
                    log_spacing = (log_inf - log_min) / 4  # 4 intervals for 5 ticks
                    
                    # Generate 5 equally spaced ticks in log space (4 finite + 1 infinity)
                    tick_values = [10 ** (log_min + i * log_spacing) for i in range(5)]
                    tick_labels = [str(int(c)) for c in tick_values[:4]] + ['$\infty$']
                    
                    # Set the last tick value to actual infinity position
                    tick_values[4] = inf_value
                    
                    # Set x-axis limits to show all ticks with some padding
                    ax.set_xlim(min_c * 0.5, inf_value * 1.5)
                else:
                    log_min = np.log10(min_c)
                    log_max = np.log10(max_c)
                    tick_values = np.logspace(log_min, log_max, 5)
                    tick_labels = [str(int(c)) for c in tick_values]
                
                ax.set_xticks(tick_values)
                # Use FixedFormatter to ensure our custom labels are preserved
                from matplotlib.ticker import FixedFormatter
                ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
            elif has_infinity_models and infinity_complexity.size > 0:
                # Only infinity models exist (no finite complexities)
                # Just show the infinity value with infinity label
                inf_value = infinity_complexity[0]
                ax.set_xticks([inf_value])
                from matplotlib.ticker import FixedFormatter
                ax.xaxis.set_major_formatter(FixedFormatter(['$\infty$']))
                ax.set_xlim(inf_value * 0.5, inf_value * 1.5)
        
        ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
        ax.minorticks_off()
        ax.grid(True, alpha=0.3, zorder=0)

    # Hide empty subplots and find the rightmost empty subplot for legend
    legend_ax = None
    
    # Hide unused subplots in first row (MAE datasets)
    for col in range(len(found_regression_datasets), n_cols):
        axes[0][col].set_visible(False)
        # Use the rightmost empty subplot in first row for legend if available
        if col == n_cols - 1:
            legend_ax = axes[0][col]
    
    # Hide unused subplots in second row (1-accuracy datasets)
    for col in range(len(classification_datasets), n_cols):
        axes[1][col].set_visible(False)
        # Use the rightmost empty subplot in second row for legend if first row doesn't have one
        if legend_ax is None and col == n_cols - 1:
            legend_ax = axes[1][col]

    # Create filtered styles based on what was actually plotted
    filtered_styles = {}
    for name in all_plotted_styles:
        if name in model_styles:
            filtered_styles[name] = model_styles[name]

    # Create custom legend handles
    custom_handles = []
    for name, style in filtered_styles.items():
        if name in ['cem', 'blackbox']:
            # Use line handle for cem and blackbox
            custom_handles.append(plt.Line2D([0], [0], color=style['color'], linestyle=':', linewidth=2, label=style['name']))
        else:
            # Use marker handle for other models
            custom_handles.append(plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=(style['size'])+10, label=style['name'], markeredgewidth=0.5, markeredgecolor='black'))

    # Place legend in empty subplot if available, otherwise below plots
    if legend_ax is not None:
        legend_ax.set_visible(True)
        legend_ax.axis('off')  # Hide axes
        legend_ax.legend(handles=custom_handles, loc='center', ncol=1, fontsize=tick_font['size'], frameon=True)
    else:
        # Create a single legend below the plots with one model per column
        fig.legend(handles=custom_handles, loc='lower center', ncol=len(custom_handles), fontsize=tick_font['size'], frameon=True, bbox_to_anchor=(0.5, -0.15), columnspacing=1.0, handletextpad=0.5)

    plt.tight_layout()
    
    plt.savefig(os.path.join(result_figs, f'pareto_front_{complexity_type}.pdf'))

#############################################
########## Latex Table Creation Utils #######
##############################################

def create_latex_tables_from_csv(csv_file_path, output_dir='tabs'):
    """
    Creates two LaTeX tables from a CSV file with performance metrics.
    
    Table 1: Task Performance (mean_task ± se_task)
    Table 2: Concept Performance (mean_concept ± se_concept)
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file containing the performance data
    output_dir : str
        Directory where the LaTeX tables will be saved
        
    Returns:
    --------
    tuple : (task_latex, concept_latex)
        Two strings containing the LaTeX table code
    """
    import pandas as pd
    import numpy as np
    import os
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get unique datasets and models
    datasets = df['dataset'].unique()
    models = df['model'].unique()
    
    # Calculate se_task and se_concept if not already present
    # Note: Since we only have one row per model/dataset, se values should already be in the CSV
    # If std_task exists but se_task doesn't, we'd need num_seeds to calculate it
    
    # Create pivot tables for task and concept metrics
    # Task table: mean_task ± se_task
    task_table = []
    concept_table = []
    
    for model in models:
        task_row = [model]
        concept_row = [model]
        
        for dataset in datasets:
            # Filter data for this model and dataset
            data = df[(df['model'] == model) & (df['dataset'] == dataset)]
            
            if len(data) > 0:
                row = data.iloc[0]
                
                # Task performance
                mean_task = row['mean_task']
                se_task = row.get('se_task', np.nan)
                
                if pd.notna(mean_task):
                    if pd.notna(se_task):
                        task_cell = f"${mean_task:.2f} \\pm {se_task:.2f}$"
                    else:
                        task_cell = f"${mean_task:.2f}$"
                else:
                    task_cell = "-"
                
                task_row.append(task_cell)
                
                # Concept performance
                mean_concept = row.get('mean_concept', np.nan)
                se_concept = row.get('se_concept', np.nan)
                
                if pd.notna(mean_concept):
                    if pd.notna(se_concept):
                        concept_cell = f"${mean_concept:.2f} \\pm {se_concept:.2f}$"
                    else:
                        concept_cell = f"${mean_concept:.2f}$"
                else:
                    concept_cell = "-"
                
                concept_row.append(concept_cell)
            else:
                task_row.append("-")
                concept_row.append("-")
        
        task_table.append(task_row)
        concept_table.append(concept_row)
    
    # Create LaTeX table strings
    n_datasets = len(datasets)
    col_format = "l" + "c" * n_datasets
    
    # Format dataset names using get_df_name function
    dataset_names = [get_df_name(d) for d in datasets]
    
    # Task performance table
    task_latex = "\\begin{table}[h]\n"
    task_latex += "\\centering\n"
    task_latex += f"\\begin{{tabular}}{{{col_format}}}\n"
    task_latex += "\\hline\n"
    # Be sure dataset_names do not contains nan, if so, replace with 'NaN'  
    dataset_names = [name if pd.notna(name) else 'NaN' for name in dataset_names]
    task_latex += "Model & " + " & ".join(dataset_names) + " \\\\\n"
    task_latex += "\\hline\n"
    
    for row in task_table:
        task_latex += " & ".join(row) + " \\\\\n"
    
    task_latex += "\\hline\n"
    task_latex += "\\end{tabular}\n"
    task_latex += "\\caption{Task Performance: $\\text{mean\\_task} \\pm \\text{se\\_task}$ (MAE for regression tasks)}\n"
    task_latex += "\\label{tab:task_performance}\n"
    task_latex += "\\end{table}\n"
    
    # Concept performance table
    concept_latex = "\\begin{table}[h]\n"
    concept_latex += "\\centering\n"
    concept_latex += f"\\begin{{tabular}}{{{col_format}}}\n"
    concept_latex += "\\hline\n"
    concept_latex += "Model & " + " & ".join(dataset_names) + " \\\\\n"
    concept_latex += "\\hline\n"
    
    for row in concept_table:
        concept_latex += " & ".join(row) + " \\\\\n"
    
    concept_latex += "\\hline\n"
    concept_latex += "\\end{tabular}\n"
    concept_latex += "\\caption{Concept Performance: $\\text{mean\\_concept} \\pm \\text{se\\_concept}$ (MAE for regression tasks)}\n"
    concept_latex += "\\label{tab:concept_performance}\n"
    concept_latex += "\\end{table}\n"
    
    # Save to files
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'task_performance_table.tex'), 'w') as f:
        f.write(task_latex)
    
    with open(os.path.join(output_dir, 'concept_performance_table.tex'), 'w') as f:
        f.write(concept_latex)



#######################################
##### Symbolic regression results #####
#######################################

def show_symbolic_regression_results(
    results_df,
    custom_order,
    table_path,):
    """
    Generate table with average and uncertainty for the Symbolic regression ablation.
    """

    table_dir = table_path + '/sr_ablation'
    os.makedirs(table_dir, exist_ok=True)

    performance = compute_avg_and_uncertainty(results_df, custom_order)

    # save the performance as csv
    performance.to_csv(f'{table_dir}/sr_ablation_performance.csv', index=False)

    # save the tex table
    create_latex_tables_from_csv(f'{table_dir}/sr_ablation_performance.csv', output_dir=table_dir)


def compute_ted_metrics_for_sr_ablation(paths, MEMORY_MODELS_LIST=['linear_symbolic_cbm', 'kan_symbolic_cbm','prior_symbolic_cbm']):
    """Compute Tree Edit Distance (TED) metrics for symbolic regression ablation experiments."""
    
    import dill
    import pickle
    try:
        from scipy.optimize import linear_sum_assignment
        use_scipy = True
    except Exception as e:
        print(f"Warning: Could not import scipy.optimize.linear_sum_assignment: {e}")
        print("Using greedy assignment algorithm instead")
        use_scipy = False
    from src.utils.ted import sympy_to_tree, ted_weighted, make_costs
    from sympy import sympify
    
    def greedy_assignment(cost_matrix):
        """Greedy assignment algorithm as fallback when scipy fails."""
        n_rows, n_cols = cost_matrix.shape
        row_ind = []
        col_ind = []
        available_cols = set(range(n_cols))
        
        # For each row, assign to the best available column
        for i in range(n_rows):
            if not available_cols:
                break
            best_col = min(available_cols, key=lambda j: cost_matrix[i, j])
            row_ind.append(i)
            col_ind.append(best_col)
            available_cols.remove(best_col)
        
        return np.array(row_ind), np.array(col_ind)
    
    # Collect all experiment paths
    exps_path = []
    for path in paths:
        experiment_dir = os.listdir(path)
        for exp in experiment_dir:
            exps_path += [os.path.join(path, exp, e) for e in os.listdir(os.path.join(path, exp)) if 'multirun' not in e]
    
    results = []
    
    # Group experiments by (dataset, seed) to find corresponding prior model
    exp_groups = {}
    for exp_path in exps_path:
        try:
            config_file = os.path.join(exp_path, '.hydra/config.yaml')
            if not os.path.exists(config_file):
                continue
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            dataset_name = config['dataset']['metadata']['name']
            model_name = config['model']['metadata']['name']
            seed = config['seed']
            
            key = (dataset_name, seed)
            if key not in exp_groups:
                exp_groups[key] = {}
            exp_groups[key][model_name] = exp_path
        except Exception as e:
            continue
    
    # Process each experiment group
    for (dataset_name, seed), models in tqdm(exp_groups.items(), desc="Processing experiment groups"):
        # Find the prior_symbolic_cbm model for this dataset/seed (ground truth)
        prior_model_path = models.get('prior_symbolic_cbm')
        
        if prior_model_path is None:
            print(f"Warning: No prior_symbolic_cbm found for dataset={dataset_name}, seed={seed}")
            continue
        
        # Load true equations from prior model's predictions CSV
        prior_predictions_file = os.path.join(prior_model_path, 'logs/experiment_metrics/test_predictions_per_sample.csv')
        if not os.path.exists(prior_predictions_file):
            print(f"Warning: No predictions file found for prior model at {prior_model_path}")
            continue
        
        try:
            df_prior = pd.read_csv(prior_predictions_file)
            
            if 'equation' not in df_prior.columns:
                print(f"Warning: 'equation' column not found in prior predictions")
                continue
            
            # Get concept variable names
            prior_vars = [x.replace('c_pred_','') for x in df_prior.columns if 'c_pred' in x]
            
            # Get unique equations from prior model
            true_equation_strs = []
            for eq_str in df_prior['equation'].unique():
                if pd.notna(eq_str) and eq_str != '':
                    # Remove target prefix if present
                    if ':' in eq_str:
                        eq_str = eq_str.split(':')[1].strip()
                    true_equation_strs.append(eq_str)
            
            if not true_equation_strs:
                print(f"Warning: No equations found in prior model for dataset={dataset_name}, seed={seed}")
                continue
            
            # Convert true equations to trees
            weight_fn, rename_fn = make_costs()
            true_trees = []
            for eq_str in true_equation_strs:
                try:
                    equation = sympify(eq_str, locals={v: sympify(v) for v in prior_vars})
                    tree = sympy_to_tree(equation, canonicalize_commutative=True, enforce_mul_for_add_terms=True)
                    true_trees.append(tree)
                except Exception as e:
                    print(f"Error converting true equation '{eq_str}' to tree: {e}")
                    continue
            
            if not true_trees:
                print(f"Warning: Could not convert any true equations to trees for dataset={dataset_name}, seed={seed}")
                continue
            
        except Exception as e:
            print(f"Error loading prior equations from {prior_predictions_file}: {e}")
            continue
        
        # Now process models in MEMORY_MODELS_LIST for this dataset/seed
        for model_name, exp_path in models.items():
            # Only process models in the specified list
            if model_name not in MEMORY_MODELS_LIST:
                continue
            
            try:
                # Load learned equations from predictions CSV
                predictions_file = os.path.join(exp_path, 'logs/experiment_metrics/test_predictions_per_sample.csv')
                if not os.path.exists(predictions_file):
                    print(f"No predictions file found for {exp_path}")
                    continue
                
                df_pred = pd.read_csv(predictions_file)
                
                if 'equation' not in df_pred.columns:
                    print(f"Warning: 'equation' column not found in predictions for {model_name}")
                    continue
                
                # Get concept variable names
                vars = [x.replace('c_pred_','') for x in df_pred.columns if 'c_pred' in x]
                
                # Get unique learned equations
                learned_equation_strs = []
                for eq_str in df_pred['equation'].unique():
                    if pd.notna(eq_str) and eq_str != '':
                        # Remove target prefix if present
                        if ':' in eq_str:
                            eq_str = eq_str.split(':')[1].strip()
                        learned_equation_strs.append(eq_str)
                
                if not learned_equation_strs:
                    print(f"No learned equations found for {exp_path}")
                    continue
                
                # Convert learned equations to trees
                learned_trees = []
                for eq_str in learned_equation_strs:
                    try:
                        equation = sympify(eq_str, locals={v: sympify(v) for v in vars})
                        tree = sympy_to_tree(equation, canonicalize_commutative=True, enforce_mul_for_add_terms=True)
                        learned_trees.append(tree)
                    except Exception as e:
                        print(f"Error converting learned equation '{eq_str}' to tree: {e}")
                        continue
                
                if not learned_trees:
                    print(f"Could not convert any learned equations to trees for {exp_path}")
                    continue
                
                # Compute TED matrix: rows = learned equations, cols = true equations
                n_learned = len(learned_trees)
                n_true = len(true_trees)
                ted_matrix = np.zeros((n_learned, n_true))
                
                for i, learned_tree in enumerate(learned_trees):
                    for j, true_tree in enumerate(true_trees):
                        try:
                            ted = ted_weighted(learned_tree, true_tree, weight_fn, rename_fn)
                            ted_matrix[i, j] = ted
                        except Exception as e:
                            print(f"Error computing TED: {e}")
                            ted_matrix[i, j] = np.inf
                
                # Optimal assignment: minimize total TED
                # If dimensions don't match, pad with high cost
                if n_learned != n_true:
                    max_dim = max(n_learned, n_true)
                    padded_matrix = np.full((max_dim, max_dim), np.max(ted_matrix) * 10)
                    padded_matrix[:n_learned, :n_true] = ted_matrix
                    if use_scipy:
                        row_ind, col_ind = linear_sum_assignment(padded_matrix)
                    else:
                        row_ind, col_ind = greedy_assignment(padded_matrix)
                    # Filter out padded assignments
                    valid_mask = (row_ind < n_learned) & (col_ind < n_true)
                    row_ind = row_ind[valid_mask]
                    col_ind = col_ind[valid_mask]
                else:
                    if use_scipy:
                        row_ind, col_ind = linear_sum_assignment(ted_matrix)
                    else:
                        row_ind, col_ind = greedy_assignment(ted_matrix)
                
                # Compute average TED for the optimal assignment
                assigned_teds = [ted_matrix[i, j] for i, j in zip(row_ind, col_ind)]
                avg_ted = np.mean(assigned_teds) if assigned_teds else np.nan
                
                results.append({
                    'dataset': dataset_name,
                    'model': model_name,
                    'seed': seed,
                    'avg_ted': avg_ted,
                    'n_learned': n_learned,
                    'n_true': n_true
                })
                
            except Exception as e:
                print(f"Error processing experiment {exp_path}: {e}")
                continue
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Create equations dataframe for a randomly selected seed
    equations_data = []
    
    if len(results_df) > 0:
        # Get all available seeds
        available_seeds = results_df['seed'].unique()
        
        if len(available_seeds) > 0:
            # Randomly select one seed
            selected_seed = np.random.choice(available_seeds)
            print(f"\nSelected seed {selected_seed} for equation collection")
            
            # Collect equations for the selected seed
            for (dataset_name, seed), models in exp_groups.items():
                if seed != selected_seed:
                    continue
                
                # Process models in MEMORY_MODELS_LIST
                for model_name, exp_path in models.items():
                    if model_name not in MEMORY_MODELS_LIST:
                        continue
                    
                    try:
                        # Load learned equations from predictions CSV
                        predictions_file = os.path.join(exp_path, 'logs/experiment_metrics/test_predictions_per_sample.csv')
                        if not os.path.exists(predictions_file):
                            continue
                        
                        df_pred = pd.read_csv(predictions_file)
                        
                        if 'equation' not in df_pred.columns:
                            continue
                        
                        # Get unique learned equations
                        for eq_str in df_pred['equation'].unique():
                            if pd.notna(eq_str) and eq_str != '':
                                # Remove target prefix if present
                                if ':' in eq_str:
                                    cleaned_eq = eq_str.split(':')[1].strip()
                                else:
                                    cleaned_eq = eq_str
                                
                                equations_data.append({
                                    'model': model_name,
                                    'dataset': dataset_name,
                                    'equation': cleaned_eq
                                })
                    
                    except Exception as e:
                        print(f"Error collecting equations from {exp_path}: {e}")
                        continue
    
    equations_df = pd.DataFrame(equations_data)
    
    return results_df, equations_df
