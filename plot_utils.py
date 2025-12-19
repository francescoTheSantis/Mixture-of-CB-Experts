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

#########################################
######### Data Extraction ###############
#########################################

def get_exp_from_path(paths):
    # Collect all the experiments in the given paths
    exps_path = []
    lmr_paths = []
    for path in paths:
        experiment_dir = os.listdir(path)
        for exp in experiment_dir:
            exps_path += [os.path.join(path, exp, e) for e in os.listdir(os.path.join(path, exp)) if 'multirun' not in e]


    performance = pd.DataFrame()

    # Iterate over all the experiments and collect the performance metrics and the config
    for exp in exps_path:
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

def get_intervention_from_path(paths, filtered_exps=None):
    performance = pd.DataFrame()

    # Collect all the experiments in the given paths
    exps_path = []
    lmr_paths = []
    
    for path in paths:
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

    if filtered_exps is None:
        return performance
    else:
        # Keep only the experiments in filtered_exps
        filtered_performance = performance.merge(filtered_exps, on=['dataset', 'model', 'memory_size'], how='inner')
        # If the model belong to cmb_linear, licem, dcr, cem, blackbox, add them to performance
        memoryless_models = performance[performance['model'].isin(['cmb_linear', 'licem', 'dcr', 'cem', 'blackbox'])]
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
        performance_acc = performance.dropna(subset=['task_acc']).groupby(['dataset', 'memory_size', 'model']).agg(
            mean_task=('task_acc', 'mean'),
            std_task=('task_acc', 'std'),
            mean_concept=('concept_acc', 'mean'),
            std_concept=('concept_acc', 'std'),
            task_type=('task_type', 'first')
        ).reset_index()
        performance_acc['metric_type'] = 'accuracy'
        
        performance_mse = performance.dropna(subset=['task_mae']).groupby(['dataset', 'memory_size', 'model']).agg(
            mean_task=('task_mae', 'mean'),
            std_task=('task_mae', 'std'),
            mean_concept=('concept_mae', 'mean'),
            std_concept=('concept_mae', 'std'),
            task_type=('task_type', 'first')
        ).reset_index()
        performance_mse['metric_type'] = 'mae'
        
        performance = pd.concat([performance_acc, performance_mse], ignore_index=True)
    else:
        # Fallback to original logic
        task_col = 'task_acc' if 'task_acc' in performance.columns else 'task_mae'
        concept_col = 'concept_acc' if 'concept_acc' in performance.columns else 'concept_mae'
        performance = performance.groupby(['dataset', 'memory_size', 'model']).agg(
            mean_task=(task_col, 'mean'),
            std_task=(task_col, 'std'),
            mean_concept=(concept_col, 'mean'),
            std_concept=(concept_col, 'std'),
            task_type=('task_type', 'first')
        ).reset_index()
        performance['metric_type'] = 'accuracy' if 'acc' in task_col else 'mae'

    # instead of the std compute the standard error at 95% confidence
    performance['se_task'] = 1.96 * performance['std_task'] / np.sqrt(num_seeds)
    performance['se_concept'] = 1.96 * performance['std_concept'] / np.sqrt(num_seeds)

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

    # Separate datasets by task type
    classification_datasets = [d for d in unique_datasets if d not in regression_datasets]
    found_regression_datasets = [d for d in unique_datasets if d in regression_datasets]
    
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
        # Both types exist - organize with classification first, then regression
        organized_datasets = classification_datasets + found_regression_datasets
        n_cols = max(len(classification_datasets), len(found_regression_datasets))
        n_rows = 2  # Force 2 rows: classification on first row, regression on second
    
    # sort the organized_datasets according to custom_order
    organized_datasets = [d for d in custom_order if d in organized_datasets]

    n_datasets = len(organized_datasets)

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
            # Both types - classification on row 0, regression on row 1
            if dataset in classification_datasets:
                row = 0
                col = classification_datasets.index(dataset)
            else:
                row = 1
                col = found_regression_datasets.index(dataset)
        
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
            
            # Convert accuracy to 1-accuracy (error rate) for classification tasks
            if dataset not in found_regression_datasets:
                grouped_data['mean_metric'] = 100 - grouped_data['mean_metric']
            
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
            # Both types - show xlabel on row 1 (regression row)
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
                if dataset not in found_regression_datasets:
                    ylabel = '$\Delta$ (Error Rate)' if relative_accuracy else 'Error Rate'
                    ax.set_ylabel(ylabel, fontsize=label_font['size'])
                else:
                    ylabel = '$\Delta$ MAE' if relative_accuracy else 'MAE'
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
        
        # Hide unused subplots in first row
        for col in range(len(classification_datasets), n_cols):
            axes[0][col].set_visible(False)
        
        # Hide unused subplots in second row and check for legend placement
        for col in range(len(found_regression_datasets), n_cols):
            axes[1][col].set_visible(False)
            # If this is the last (rightmost) empty subplot and total datasets is odd
            if col == n_cols - 1 and total_datasets % 2 == 1:
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
            bbox_to_anchor=(0.5, -0.1),
            columnspacing=1.0,
            handletextpad=0.5
        )

    plt.tight_layout()
    str_store = str(unique_noises[0]).replace('.', '')
    suffix = 'relative_accuracy_difference' if relative_accuracy else 'absolute_accuracy'
    prefix = out_dir if out_dir is not None else 'figs'
    os.makedirs(f'{prefix}/intervention/{suffix}', exist_ok=True)
    plt.savefig(f'{prefix}/intervention/{suffix}/{str_store}.pdf')
    plt.show()

def plot_intervention_memory_results(df, 
                                    p_int=0.80,  # Fixed p_int value
                                    metric='accuracy', 
                                    unique_noises=[0.0], 
                                    title_font=None, 
                                    label_font=None, 
                                    tick_font=None, 
                                    legend_font=None,
                                    custom_order=None,
                                    model_styles=None,
                                    relative_accuracy=False,
                                    n_mechanisms=None):
    unique_datasets = custom_order
    

    # From df eliminate blackbox and for the models in: ['licem', 'dcr', 'cem']
    # set the memory to 500
    # df = df[~df['model'].isin(['blackbox'])]
    # Set cmb_linear's memory_size to 1 and licem/dcr's memory_size to 500
    df.loc[df['model'].isin(['cmb_linear', 'cem']), 'memory_size'] = 1
    df.loc[df['model'].isin(['licem', 'dcr']), 'memory_size'] = 500

    # NOTE: keep in mind to update this list if you add new regression datasets
    regression_datasets = ['mnist_arithmetic', 'dsprites_simple', 'dsprites_complex', 'cebab', 'pendulum', 'mawps']
    
    # Separate datasets by task type
    classification_datasets = [d for d in unique_datasets if d not in regression_datasets]
    regression_datasets = [d for d in unique_datasets if d in regression_datasets]
    
    # Organize datasets with regression first, then classification (MAE first row, 1-accuracy second row)
    organized_datasets = list(regression_datasets) + list(classification_datasets)
    
    n_datasets = len(organized_datasets)
    n_cols = max(len(classification_datasets), len(regression_datasets))
    n_rows = 2  # Force 2 rows: MAE on first row, 1-accuracy on second

    # Save the df as a csv for future reference
    df.to_csv(f'tabs/intervention_results.csv')

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows), sharex=False, sharey=False)

    # Handle case where we have only one subplot
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, dataset in enumerate(organized_datasets):
        # Determine row based on task type
        if dataset in regression_datasets:
            row = 0
            col = regression_datasets.index(dataset)
        else:
            row = 1
            col = classification_datasets.index(dataset)
        
        if n_rows == 1:
            ax = axes[col] if n_cols > 1 else axes[0]
        else:
            ax = axes[row][col] if n_cols > 1 else axes[row][0] if col == 0 else axes[row][col]
        
        for noise in unique_noises:
            # Filter data for the specific p_int, noise, and dataset
            data = df[(df['noise'] == noise) & (df['dataset'] == dataset) & (df['p_int'] == p_int)]
            if dataset in regression_datasets:
                metric = 'mae'
            else:
                metric = 'accuracy'
            grouped_data = data.groupby(['memory_size', 'model']).agg(
                mean_metric=(metric, 'mean'),
                std_metric=(metric, 'std')
            ).reset_index().fillna(0)
            
            # Calculate number of seeds for standard error
            num_seeds = data.groupby(['memory_size', 'model']).size().reset_index(name='n_seeds')
            grouped_data = grouped_data.merge(num_seeds, on=['memory_size', 'model'])
            grouped_data['se_metric'] = 1.96 * grouped_data['std_metric'] / np.sqrt(grouped_data['n_seeds'])
            
            # Convert accuracy to 1-accuracy (error rate) for classification tasks
            if dataset not in regression_datasets:
                grouped_data['mean_metric'] = 100 - grouped_data['mean_metric']
            
            if relative_accuracy:
                # Calculate relative accuracy for each model (baseline should be p_int=0)
                baseline_data = df[(df['noise'] == noise) & (df['dataset'] == dataset) & (df['p_int'] == 0)]
                baseline_grouped = baseline_data.groupby(['memory_size', 'model']).agg(
                    baseline_metric=(metric, 'mean')
                ).reset_index()
                
                if dataset not in regression_datasets:
                    baseline_grouped['baseline_metric'] = 1 - baseline_grouped['baseline_metric']
                
                grouped_data = grouped_data.merge(baseline_grouped, on=['memory_size', 'model'], how='left')
                grouped_data['mean_metric'] = grouped_data['mean_metric'] - grouped_data['baseline_metric'].fillna(0)
            
            for model in grouped_data['model'].unique():
                model_data = grouped_data[grouped_data['model'] == model]
                style = model_styles.get(model, {'marker': 'o', 'color': 'black', 'size': 10, 'name': model})
                
                # Check if this model has unique memory size (only one data point)
                has_unique_memory_size = len(model_data) == 1
                
                if has_unique_memory_size:
                    # Show uncertainty as vertical error bars for single points
                    ax.errorbar(
                        model_data['memory_size'], 
                        model_data['mean_metric'],
                        yerr=model_data['se_metric'],
                        label=style['name'],
                        marker=style['marker'], 
                        color=style['color'], 
                        markersize=style['size'],
                        markeredgecolor='black',
                        markeredgewidth=0.1,
                        capsize=3,
                        capthick=1.5,
                        alpha=0.8,
                        linestyle='none'  # No line for single points
                    )
                else:
                    # Plot the line and markers
                    ax.plot(model_data['memory_size'], model_data['mean_metric'],
                            marker=style['marker'], 
                            color=style['color'], 
                            markersize=style['size'], 
                            label=style['name'],
                            markeredgecolor='black',
                            markeredgewidth=0.1,
                            alpha=0.8,
                            linestyle='-')
                    
                    # Add shaded area for uncertainty
                    ax.fill_between(model_data['memory_size'],
                                    model_data['mean_metric'] - model_data['se_metric'],
                                    model_data['mean_metric'] + model_data['se_metric'],
                                    color=style['color'],
                                    alpha=0.2)
        
        # Get distinct memory sizes for this dataset
        distinct_memory_sizes = sorted(grouped_data['memory_size'].unique()) if not grouped_data.empty else []
        
        if distinct_memory_sizes:
            # Set x-axis to log scale first
            ax.set_xscale('log')
            # Set ticks to actual values
            ax.set_xticks(distinct_memory_sizes)
            # Force display of actual values instead of scientific notation
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
            # Create custom labels
            x_labels = []
            for size in distinct_memory_sizes:
                if size == 500:
                    x_labels.append('$\infty$')
                else:
                    label = str(int(size))
                    # Make label bold if this dataset is in n_mechanisms and size matches
                    if n_mechanisms and dataset in n_mechanisms and size == n_mechanisms[dataset]:
                        label = f'$\mathbf{{{label}}}$'
                    x_labels.append(label)
            ax.set_xticklabels(x_labels)
        
        # Show xlabel only for the last row
        if row == 1:
            ax.set_xlabel('Memory Size', fontsize=label_font['size'])
        
        ax.set_title(f'{get_df_name(dataset)}', fontsize=label_font['size'])

        # Set ylabel only for the leftmost subplot in each row
        if col == 0:
            if dataset in regression_datasets:
                ylabel = '$\Delta$ MAE' if relative_accuracy else 'MAE'
                ax.set_ylabel(ylabel, fontsize=label_font['size'])
            else:
                ylabel = '$\Delta$ (Error Rate)' if relative_accuracy else 'Error Rate'
                ax.set_ylabel(ylabel, fontsize=label_font['size'])
        
        ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
        ax.minorticks_off()
        ax.grid(True, alpha=0.3, zorder=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # Hide empty subplots and find the rightmost empty subplot for legend
    legend_ax = None
    
    # Hide unused subplots in first row (MAE datasets)
    for col in range(len(regression_datasets), n_cols):
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
    
    # Filter the style according to the models' names which are present in the df
    filtered_styles = {name: style for name, style in model_styles.items() if name in df['model'].values}

    # Create custom legend handles
    custom_handles = [plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=style['size']+10, label=style['name'], markeredgewidth=0.5, markeredgecolor='black') for style in filtered_styles.values()]

    # Place legend in empty subplot if available, otherwise below plots
    if legend_ax is not None:
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
            bbox_to_anchor=(0.5, -0.1),
            columnspacing=1.0,
            handletextpad=0.5
        )

    plt.tight_layout()
    str_store = str(p_int).replace('.', '')
    suffix = 'relative_accuracy_difference' if relative_accuracy else 'absolute_accuracy'
    os.makedirs(f'figs/intervention_memory/{suffix}', exist_ok=True)
    plt.savefig(f'figs/intervention_memory/{suffix}/interventions_pint_{str_store}.pdf')
    plt.show()

###########################################
######### Plotting Memory Ablation ########
###########################################

def plot_memory_ablation(performance, model_styles, title_font, label_font, tick_font, custom_order):

    performance = compute_avg_and_uncertainty(performance, custom_order)

    # Separate datasets by task type
    classification_datasets = performance[performance['task_type'] == 'classification']['dataset'].unique()
    regression_datasets = performance[performance['task_type'] == 'regression']['dataset'].unique()
    
    # Organize datasets with MAE first, then 1-accuracy
    organized_datasets = list(regression_datasets) + list(classification_datasets)
    
    n_datasets = len(organized_datasets)
    n_cols = max(len(classification_datasets), len(regression_datasets))  
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
        if dataset in regression_datasets:
            row = 0
            col = list(regression_datasets).index(dataset)
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
    for col in range(len(regression_datasets), n_cols):
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
    plt.savefig(os.path.join(os.environ.get("RESULT_FIGS"), 'memory_ablation.pdf'))

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
    plt.savefig(os.path.join(os.environ.get("RESULT_FIGS"), 'concept_size_ablation.pdf'), bbox_inches='tight')



############################################
############## Pareto plots ################
############################################

def filter_pareto_models(df: pd.DataFrame, fixed_memory: dict = None, custom_order: list = None, delta_threshold: float = 0.1) -> pd.DataFrame:
    """
    Filters a dataframe of model results to keep only the best memory_size per model/dataset/seed
    using Pareto optimality with delta threshold:
        - For classification: maximize accuracy, minimize memory_size
        - For regression: minimize MAE, minimize memory_size

    If `fixed_memory` is provided (dict mapping dataset -> memory_size), then for those datasets
    the given memory_size is enforced.
    
    If delta_threshold is provided, selects the smallest memory_size where the performance
    difference with the next memory_size is below the threshold.
    """

    df = compute_avg_and_uncertainty(df, custom_order)

    fixed_memory = fixed_memory or {}

    def pareto_front(group: pd.DataFrame) -> pd.DataFrame:
        dataset = group["dataset"].iloc[0]
        task_type = group["task_type"].iloc[0]

        # If dataset has fixed memory size, pick that one if available
        if dataset in fixed_memory:
            mem_size = fixed_memory[dataset]
            fixed_group = group[group["memory_size"] == mem_size]
            if not fixed_group.empty:
                return fixed_group.iloc[0].to_frame().T
            else:
                group = group.iloc[0].to_frame().T
                return group

        # Sort by memory_size to compare consecutive values
        group = group.sort_values("memory_size")
        
        if task_type == "classification":
            # Higher accuracy is better, lower memory_size is better
            best_acc = group["mean_task"].max()
            best_candidates = group[group["mean_task"] == best_acc]
            
            if len(best_candidates) == 1:
                return best_candidates.iloc[0].to_frame().T
            
            # Check delta threshold among best candidates
            best_candidates = best_candidates.sort_values("memory_size")
            for i in range(len(best_candidates) - 1):
                current = best_candidates.iloc[i]
                next_val = best_candidates.iloc[i + 1]
                delta = next_val["mean_task"] - current["mean_task"]
                if delta <= delta_threshold:
                    return current.to_frame().T
            
            # If no delta below threshold, return smallest memory_size
            return best_candidates.loc[best_candidates["memory_size"].idxmin()].to_frame().T

        elif task_type == "regression":
            # Lower MAE is better, lower memory_size is better
            best_mae = group["mean_task"].min()
            best_candidates = group[group["mean_task"] == best_mae]
            
            if len(best_candidates) == 1:
                return best_candidates.iloc[0].to_frame().T
            
            # Check delta threshold among best candidates
            best_candidates = best_candidates.sort_values("memory_size")
            for i in range(len(best_candidates) - 1):
                current = best_candidates.iloc[i]
                next_val = best_candidates.iloc[i + 1]
                delta = current["mean_task"] - next_val["mean_task"]
                if delta <= delta_threshold:
                    return current.to_frame().T
            
            # If no delta below threshold, return smallest memory_size
            return best_candidates.loc[best_candidates["memory_size"].idxmin()].to_frame().T
        else:
            return group

    # Apply per dataset-model-seed
    filtered = df.groupby(["dataset", "model"], group_keys=False).apply(pareto_front)

    # Return a dataframe containing:
    # - dataset
    # - model
    # - memory_size
    filtered = filtered[["dataset", "model", "memory_size"]]

    return filtered.reset_index(drop=True)

def plot_pareto_front(performance, model_styles, title_font, label_font, tick_font, custom_order):

    performance = compute_avg_and_uncertainty(performance, custom_order)

    # save the performance as csv
    performance.to_csv(os.path.join(os.environ.get("TABLE_PATH"), 'memory_ablation_performance.csv'), index=False)

    # Define operational complexity for each model
    oc = { 
        'licem': 2,
        'dcr': 3,
        'cmr': 3,
        'linear_symbolic_cbm': 2,
        'kan_symbolic_cbm': 7,
    }

    # Filter out dcr and cmr for cub200, awa2, and cifar10 datasets
    performance = performance[~((performance['dataset'].isin(['cub', 'awa2', 'cifar10'])) & (performance['model'].isin(['dcr', 'cmr'])))]

    # Separate datasets by task type
    classification_datasets = performance[performance['task_type'] == 'classification']['dataset'].unique()
    regression_datasets = performance[performance['task_type'] == 'regression']['dataset'].unique()

    # Organize datasets with MAE first, then 1-accuracy
    organized_datasets = list(regression_datasets) + list(classification_datasets)
    
    n_datasets = len(organized_datasets)
    n_cols = max(len(classification_datasets), len(regression_datasets))  
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
        if dataset in regression_datasets:
            row = 0
            col = list(regression_datasets).index(dataset)
        else:
            row = 1
            col = list(classification_datasets).index(dataset)
        
        if n_rows == 1:
            ax = axes[col] if n_cols > 1 else axes[0]
        else:
            ax = axes[row][col] if n_cols > 1 else axes[row][0] if col == 0 else axes[row][col]
            
        data = performance[performance['dataset'] == dataset]
        metric_type = data['metric_type'].iloc[0]
        
        # Compute complexity for each model
        data = data.copy()
        data['complexity'] = data.apply(lambda row: row['memory_size'] * oc.get(row['model'], 1), axis=1)
        
        # Get distinct complexities for this dataset (excluding cem and blackbox)
        pareto_data = data[~data['model'].isin(['cem', 'blackbox'])]
        distinct_complexities = sorted(pareto_data['complexity'].unique())
        
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
                xlim = ax.get_xlim() if ax.get_xlim() != (0.0, 1.0) else (1, 1000)  # Default range if not set
                y_val = y_values.iloc[0]
                
                ax.axhline(y=y_val, color=model_styles[model]['color'], 
                            linestyle=':', linewidth=2, alpha=0.8,
                            label=model_styles[model]['name'])
                
                # Do not add uncertainty shading for blackbox and cem
                
            else:
                # Store points for Pareto front (excluding cem and blackbox)
                for i, (complexity, y_val, y_err, memory_size) in enumerate(zip(model_data['complexity'], y_values, y_errors, model_data['memory_size'])):
                    all_points.append((complexity, y_val, y_err, model, memory_size))
                
                # Plot each point with appropriate style
                for i, (complexity, y_val, y_err, memory_size) in enumerate(zip(model_data['complexity'], y_values, y_errors, model_data['memory_size'])):
                    # Use cbm_linear style if linear_symbolic_cbm has memory_size = 1
                    if model == 'linear_symbolic_cbm' and memory_size == 1:
                        plot_style = model_styles.get('cbm_linear', model_styles[model])
                        plot_label = model_styles.get('cbm_linear', {})['name'] if 'cbm_linear' in model_styles else model_styles[model]['name']
                    else:
                        plot_style = model_styles[model]
                        plot_label = model_styles[model]['name']
                    
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
            ax.set_xlabel('Model Complexity', fontdict=label_font)
        else:
            ax.set_xlabel('')
        
        # Set y-label only for the leftmost subplot in each row
        if col == 0:
            ylabel = 'MAE' if metric_type == 'mae' else 'Error Rate'
            ax.set_ylabel(ylabel, fontdict=label_font)
        
        # Set x-axis to log scale
        ax.set_xscale('log')

        # Set x-axis ticks to the actual complexity values
        if distinct_complexities:
            ax.set_xticks(distinct_complexities)
            ax.set_xticklabels([str(int(c)) for c in distinct_complexities])
        
        ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
        ax.minorticks_off()
        ax.grid(True, alpha=0.3, zorder=0)

    # Hide empty subplots and find the rightmost empty subplot for legend
    legend_ax = None
    
    # Hide unused subplots in first row (MAE datasets)
    for col in range(len(regression_datasets), n_cols):
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

    # Create filtered styles including both original models and cbm_linear for kan_symbolic_cbm with memory_size=1
    filtered_styles = {}
    for name, style in model_styles.items():
        if name in performance['model'].values:
            filtered_styles[name] = style
    
    # Add cbm_linear style if kan_symbolic_cbm appears with memory_size=1
    if 'kan_symbolic_cbm' in performance['model'].values and 'cbm_linear' in model_styles:
        if any((performance['model'] == 'kan_symbolic_cbm') & (performance['memory_size'] == 1)):
            filtered_styles['cbm_linear'] = model_styles['cbm_linear']

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
    
    plt.savefig(os.path.join(os.environ.get("RESULT_FIGS"), 'pareto_front.pdf'))


def plot_intervention_memory_pareto_results(df, 
                                    p_int=0.80,  # Fixed p_int value
                                    metric='accuracy', 
                                    unique_noises=[0.0], 
                                    title_font=None, 
                                    label_font=None, 
                                    tick_font=None, 
                                    legend_font=None,
                                    custom_order=None,
                                    model_styles=None,
                                    relative_accuracy=False,
                                    n_mechanisms=None):
    
    unique_datasets = custom_order
    
    # Define operational complexity for each model
    oc = { 
        'licem': 2,
        'dcr': 3,
        'cmr': 3,
        'linear_symbolic_cbm': 2,
        'kan_symbolic_cbm': 7,
    }
    
    # Filter out dcr and cmr for cub, awa2, and cifar10 datasets
    df = df[~((df['dataset'].isin(['cub', 'awa2', 'cifar10'])) & (df['model'].isin(['dcr', 'cmr'])))]
    
    # Multiply by 100 the accuracy columns
    df['accuracy'] = df['accuracy'] * 100

    # From df eliminate blackbox and for the models in: ['licem', 'dcr', 'cem']
    # set the memory to 500
    # df = df[~df['model'].isin(['blackbox'])]
    # Set cmb_linear's memory_size to 1 and licem/dcr's memory_size to 500
    df.loc[df['model'].isin(['cmb_linear', 'cem']), 'memory_size'] = 1
    df.loc[df['model'].isin(['licem', 'dcr']), 'memory_size'] = 500
    
    # Separate datasets by task type
    classification_datasets = [d for d in unique_datasets if d not in regression_datasets]
    regression_datasets = [d for d in unique_datasets if d in regression_datasets]
    
    # Organize datasets with regression first, then classification (MAE first row, 1-accuracy second row)
    organized_datasets = list(regression_datasets) + list(classification_datasets)
    
    n_datasets = len(organized_datasets)
    n_cols = max(len(classification_datasets), len(regression_datasets))
    n_rows = 2  # Force 2 rows: MAE on first row, 1-accuracy on second

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows), sharex=False, sharey=False)

    # Handle case where we have only one subplot
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    for idx, dataset in enumerate(organized_datasets):
        # Determine row based on task type
        if dataset in regression_datasets:
            row = 0
            col = regression_datasets.index(dataset)
        else:
            row = 1
            col = classification_datasets.index(dataset)
        
        if n_rows == 1:
            ax = axes[col] if n_cols > 1 else axes[0]
        else:
            ax = axes[row][col] if n_cols > 1 else axes[row][0] if col == 0 else axes[row][col]
        
        for noise in unique_noises:
            # Filter data for the specific p_int, noise, and dataset
            data = df[(df['noise'] == noise) & (df['dataset'] == dataset) & (df['p_int'] == p_int)]
            if dataset in regression_datasets:
                metric = 'mae'
            else:
                metric = 'accuracy'
            grouped_data = data.groupby(['memory_size', 'model']).agg(
                mean_metric=(metric, 'mean'),
                std_metric=(metric, 'std')
            ).reset_index().fillna(0)
            
            # Calculate number of seeds for standard error
            num_seeds = data.groupby(['memory_size', 'model']).size().reset_index(name='n_seeds')
            grouped_data = grouped_data.merge(num_seeds, on=['memory_size', 'model'])
            grouped_data['se_metric'] = 1.96 * grouped_data['std_metric'] / np.sqrt(grouped_data['n_seeds'])
            
            # Compute complexity for each model
            grouped_data['complexity'] = grouped_data.apply(
                lambda row: row['memory_size'] * oc.get(row['model'], 1), axis=1)
            
            # Convert accuracy to 1-accuracy (error rate) for classification tasks
            if dataset not in regression_datasets:
                grouped_data['mean_metric'] = 100 - grouped_data['mean_metric']
            
            if relative_accuracy:
                # Calculate relative accuracy for each model (baseline should be p_int=0)
                baseline_data = df[(df['noise'] == noise) & (df['dataset'] == dataset) & (df['p_int'] == 0)]
                baseline_grouped = baseline_data.groupby(['memory_size', 'model']).agg(
                    baseline_metric=(metric, 'mean')
                ).reset_index()
                
                if dataset not in regression_datasets:
                    baseline_grouped['baseline_metric'] = 100 - baseline_grouped['baseline_metric']
                
                grouped_data = grouped_data.merge(baseline_grouped, on=['memory_size', 'model'], how='left')
                grouped_data['mean_metric'] = grouped_data['mean_metric'] - grouped_data['baseline_metric'].fillna(0)
            
            # Collect points for Pareto front calculation
            all_points = []
            
            for model in grouped_data['model'].unique():
                model_data = grouped_data[grouped_data['model'] == model]
                
                # Store points for Pareto front calculation
                for i, (complexity, y_val, y_err, memory_size) in enumerate(zip(
                    model_data['complexity'], model_data['mean_metric'], 
                    model_data['se_metric'], model_data['memory_size'])):
                    all_points.append((complexity, y_val, y_err, model, memory_size))
                
                # Plot each point with appropriate style
                for i, (complexity, y_val, y_err, memory_size) in enumerate(zip(
                    model_data['complexity'], model_data['mean_metric'], 
                    model_data['se_metric'], model_data['memory_size'])):
                    
                    # Use cbm_linear style if kan_symbolic_cbm has memory_size = 1
                    if model == 'kan_symbolic_cbm' and memory_size == 1:
                        plot_style = model_styles.get('cbm_linear', model_styles[model])
                        plot_label = model_styles.get('cbm_linear', {}).get('name', model_styles[model]['name']) if 'cbm_linear' in model_styles else model_styles[model]['name']
                    else:
                        plot_style = model_styles[model]
                        plot_label = model_styles[model]['name']
                    
                    # Only add label for the first point of each unique style to avoid duplicate legends
                    label = plot_label if i == 0 else ""
                
                    # Show uncertainty as vertical error bars for all points (no connecting lines)
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
                        linestyle='none'  # No line connecting points
                    )
            
            # Calculate and draw Pareto front
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
        
        # Get distinct complexities for this dataset
        distinct_complexities = sorted(grouped_data['complexity'].unique()) if not grouped_data.empty else []
        
        if distinct_complexities:
            # Set x-axis to log scale first
            ax.set_xscale('log')
            # Set ticks to actual values
            ax.set_xticks(distinct_complexities)
            # Force display of actual values instead of scientific notation
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
            # Create custom labels showing complexity values
            x_labels = [str(int(c)) for c in distinct_complexities]
            ax.set_xticklabels(x_labels)
        
        # Show xlabel only for the last row
        if row == 1:
            ax.set_xlabel('Model Complexity', fontsize=label_font['size'])
        
        ax.set_title(f'{get_df_name(dataset)}', fontsize=label_font['size'])

        # Set ylabel only for the leftmost subplot in each row
        if col == 0:
            if dataset in regression_datasets:
                ylabel = '$\Delta$ MAE' if relative_accuracy else 'MAE'
                ax.set_ylabel(ylabel, fontsize=label_font['size'])
            else:
                ylabel = '$\Delta$ (Error Rate)' if relative_accuracy else 'Error Rate'
                ax.set_ylabel(ylabel, fontsize=label_font['size'])
        
        ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
        ax.minorticks_off()
        ax.grid(True, alpha=0.3, zorder=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # Hide empty subplots and find the rightmost empty subplot for legend
    legend_ax = None
    
    # Hide unused subplots in first row (MAE datasets)
    for col in range(len(regression_datasets), n_cols):
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
    
    # Create filtered styles including both original models and cbm_linear for kan_symbolic_cbm with memory_size=1
    filtered_styles = {}
    for name, style in model_styles.items():
        if name in df['model'].values:
            filtered_styles[name] = style

    # Add cbm_linear style if linear_symbolic_cbm appears with memory_size=1
    if 'linear_symbolic_cbm' in df['model'].values and 'cbm_linear' in model_styles:
        if any((df['model'] == 'linear_symbolic_cbm') & (df['memory_size'] == 1)):
            filtered_styles['cbm_linear'] = model_styles['cbm_linear']

    # Create custom legend handles
    custom_handles = [plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=style['size']+10, label=style['name'], markeredgewidth=0.5, markeredgecolor='black') for style in filtered_styles.values()]

    # Place legend in empty subplot if available, otherwise below plots
    if legend_ax is not None:
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
            bbox_to_anchor=(0.5, -0.1),
            columnspacing=1.0,
            handletextpad=0.5
        )

    plt.tight_layout()
    str_store = str(p_int).replace('.', '')
    suffix = 'relative_accuracy_difference' if relative_accuracy else 'absolute_accuracy'
    os.makedirs(os.path.join(os.environ.get("RESULT_FIGS"), f'intervention_memory_pareto/{suffix}'), exist_ok=True)
    plt.savefig(os.path.join(os.environ.get("RESULT_FIGS"), f'intervention_memory_pareto/{suffix}/interventions_pint_{str_store}.pdf'))
    plt.show()


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


def compute_ted_metrics_for_sr_ablation(paths):
    """
    Compute Tree Edit Distance (TED) metrics for symbolic regression ablation experiments.
    
    This function:
    - Lists all experiment directories
    - For each experiment, loads learned equations from the model's memory slots
    - Loads true equations from the corresponding prior_symbolic_cbm model's memory slots
    - Computes TED between all combinations of learned and true equations
    - Assigns memory equations to true equations via optimal matching (minimizing total TED)
    - Computes average TED for each experiment
    - Returns a CSV with dataset, model, seed, and averaged TED
    
    Args:
        path: Path to the sr_ablation output directory
        
    Returns:
        pd.DataFrame with columns: dataset, model, seed, avg_ted
    """
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
        # Find the prior_symbolic_cbm model for this dataset/seed
        prior_model_path = models.get('prior_symbolic_cbm')
        
        if prior_model_path is None:
            print(f"Warning: No prior_symbolic_cbm found for dataset={dataset_name}, seed={seed}")
            # Skip all models in this group since we need the prior as ground truth
            continue
        
        # Load true equations from prior model's memory slots
        prior_memory_dir = os.path.join(prior_model_path, 'logs/experiment_metrics/equations/memory_slots')
        if not os.path.exists(prior_memory_dir):
            print(f"Warning: No memory slots found for prior model at {prior_model_path}")
            continue
        
        prior_memory_slots = [d for d in os.listdir(prior_memory_dir) if d.startswith('memory_slot_')]
        
        true_equations = []
        for mem_slot in prior_memory_slots:
            mem_slot_dir = os.path.join(prior_memory_dir, mem_slot)
            eq_files = [f for f in os.listdir(mem_slot_dir) if f.startswith('equation_') and f.endswith('.pkl')]
            
            for eq_file in eq_files:
                eq_path = os.path.join(mem_slot_dir, eq_file)
                try:
                    with open(eq_path, 'rb') as f:
                        eq = dill.load(f)
                    true_equations.append(eq)
                except Exception as e:
                    print(f"Error loading prior equation from {eq_path}: {e}")
                    continue
        
        if not true_equations:
            print(f"Warning: No equations found in prior model for dataset={dataset_name}, seed={seed}")
            continue
        
        # Convert true equations to trees once
        weight_fn, rename_fn = make_costs()
        true_trees = []
        for eq in true_equations:
            try:
                tree = sympy_to_tree(eq, canonicalize_commutative=True, enforce_mul_for_add_terms=True)
                true_trees.append(tree)
            except Exception as e:
                print(f"Error converting true equation to tree: {e}")
                continue
        
        if not true_trees:
            continue
        
        # Now process all other models in this group
        for model_name, exp_path in models.items():
            # Process prior_symbolic_cbm as well to show TED=0 as sanity check
            
            try:
                # Find learned equations for this model
                memory_slots_dir = os.path.join(exp_path, 'logs/experiment_metrics/equations/memory_slots')
                if not os.path.exists(memory_slots_dir):
                    print(f"No memory slots directory found for {exp_path}")
                    continue
                
                memory_slots = [d for d in os.listdir(memory_slots_dir) if d.startswith('memory_slot_')]
                
                # Load all learned equations from memory slots
                learned_equations = []
                for mem_slot in memory_slots:
                    mem_slot_dir = os.path.join(memory_slots_dir, mem_slot)
                    eq_files = [f for f in os.listdir(mem_slot_dir) if f.startswith('equation_') and f.endswith('.pkl')]
                    
                    for eq_file in eq_files:
                        eq_path = os.path.join(mem_slot_dir, eq_file)
                        try:
                            with open(eq_path, 'rb') as f:
                                eq = dill.load(f)
                            learned_equations.append(eq)
                        except Exception as e:
                            print(f"Error loading equation from {eq_path}: {e}")
                            continue
                
                if not learned_equations:
                    print(f"No learned equations found for {exp_path}")
                    continue
                
                # Convert learned equations to trees
                learned_trees = []
                for eq in learned_equations:
                    try:
                        tree = sympy_to_tree(eq, canonicalize_commutative=True, enforce_mul_for_add_terms=True)
                        learned_trees.append(tree)
                    except Exception as e:
                        print(f"Error converting learned equation to tree: {e}")
                        continue
                
                if not learned_trees:
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
    return results_df


def compute_equation_complexity_for_sr_ablation(paths, n_samples=None, random_seed=None):
    """
    Compute complexity (visitation length) of learned equations for symbolic regression ablation.
    
    This function:
    - Lists all experiment directories
    - For each experiment, loads equations from test_predictions_per_sample.csv
    - Optionally samples random rows from the dataframe
    - Computes complexity (visitation length) for each equation (one per sample)
    - Averages complexity across all samples
    - Groups by (dataset, model, seed) and then averages across seeds
    - Returns a CSV with dataset, model, and averaged complexity metrics
    
    Args:
        paths: List of paths to the output directories
        n_samples: Optional number of random samples to use for complexity computation.
                   If None, all rows are used. If provided, randomly samples n_samples rows.
        random_seed: Optional random seed for reproducible sampling
        
    Returns:
        pd.DataFrame with columns: dataset, model, mean_complexity, std_complexity, n_seeds
    """
    from src.utils.complexity import compute_complexity
    from sympy import sympify
    
    # Collect all experiment paths
    exps_path = []
    for path in paths:
        experiment_dir = os.listdir(path)
        for exp in experiment_dir:
            exps_path += [os.path.join(path, exp, e) for e in os.listdir(os.path.join(path, exp)) if 'multirun' not in e]
    
    results = []
    
    # Process each experiment
    for exp_path in tqdm(exps_path, desc="Processing experiments for complexity"):
        
        try:
            # Load config to get dataset info, model, seed
            config_file = os.path.join(exp_path, '.hydra/config.yaml')
            if not os.path.exists(config_file):
                continue
                
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            dataset_name = config['dataset']['metadata']['name']
            model_name = config['model']['metadata']['name']
            seed = config['seed']
            
            # Read test_predictions_per_sample.csv
            predictions_file = os.path.join(exp_path, 'logs/experiment_metrics/test_predictions_per_sample.csv')
            if not os.path.exists(predictions_file):
                # Skip experiments without predictions file
                continue
            
            df = pd.read_csv(predictions_file)
            
            # Check if 'equation' column exists
            if 'equation' not in df.columns:
                print(f"Warning: 'equation' column not found in {predictions_file}")
                continue
            
            # Sample random rows if n_samples is specified
            if n_samples is not None and n_samples < len(df):
                df = df.sample(n=n_samples, random_state=random_seed)
            
            # Compute complexity for each equation in the dataset
            complexities = []
            for idx, row in df.iterrows():
                equation_str = row['equation']
                
                # Skip NaN or empty equations
                if pd.isna(equation_str) or equation_str == '':
                    continue
                
                # Convert string to sympy expression
                equation = sympify(equation_str)
                
                # Compute complexity
                complexity = compute_complexity(equation, metric='visitation_length')
                complexities.append(complexity)

            if not complexities:
                continue
            
            # Average complexity across all samples
            avg_complexity = np.mean(complexities)
            
            results.append({
                'dataset': dataset_name,
                'model': model_name,
                'seed': seed,
                'complexity': avg_complexity,
                'n_equations': len(complexities)
            })
            
        except Exception as e:
            print(f"Error processing experiment {exp_path}: {e}")
            continue
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        return results_df
    
    # Group by dataset and model, then average across seeds
    summary_df = results_df.groupby(['dataset', 'model']).agg(
        mean_complexity=('complexity', 'mean'),
        std_complexity=('complexity', 'std'),
        n_seeds=('seed', 'count')
    ).reset_index()
    
    return summary_df


def compare_equations_with_prior(paths, selected_seed=None):
    """
    Compare equations learned by different models with the prior model's equations.
    
    This function creates a CSV showing equations side-by-side for easy comparison.
    Each row represents one equation, showing what different models learned for it.
    
    Args:
        path: Path to the sr_ablation output directory
        selected_seed: Specific seed to use (default: uses the first available seed for each dataset)
        
    Returns:
        pd.DataFrame with columns: dataset, equation_idx, prior_equation, model1_equation, model2_equation, ...
    """
    import dill
    
    # Collect all experiment paths
    exps_path = []
    for path in paths:
        experiment_dir = os.listdir(path)
        for exp in experiment_dir:
            exps_path += [os.path.join(path, exp, e) for e in os.listdir(os.path.join(path, exp)) if 'multirun' not in e]
    
    # Group experiments by (dataset, seed)
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
    
    # For each dataset, select a seed
    dataset_seeds = {}
    for (dataset, seed), models in exp_groups.items():
        if 'prior_symbolic_cbm' not in models:
            continue  # Skip if no prior model
        
        if dataset not in dataset_seeds:
            if selected_seed is None:
                dataset_seeds[dataset] = seed  # Use first available seed
            elif seed == selected_seed:
                dataset_seeds[dataset] = seed
    
    results = []
    
    # Process each dataset
    for dataset_name, seed in tqdm(dataset_seeds.items(), desc="Comparing equations"):
        key = (dataset_name, seed)
        models = exp_groups[key]
        
        # Load prior equations first
        prior_path = models['prior_symbolic_cbm']
        prior_memory_dir = os.path.join(prior_path, 'logs/experiment_metrics/equations/memory_slots')
        
        if not os.path.exists(prior_memory_dir):
            continue
        
        prior_memory_slots = sorted([d for d in os.listdir(prior_memory_dir) if d.startswith('memory_slot_')])
        
        # Load all prior equations
        prior_equations = []
        for mem_slot in prior_memory_slots:
            mem_slot_dir = os.path.join(prior_memory_dir, mem_slot)
            eq_files = sorted([f for f in os.listdir(mem_slot_dir) if f.startswith('equation_') and f.endswith('.pkl')])
            
            for eq_file in eq_files:
                eq_path = os.path.join(mem_slot_dir, eq_file)
                try:
                    with open(eq_path, 'rb') as f:
                        eq = dill.load(f)
                    prior_equations.append(str(eq))
                except Exception as e:
                    print(f"Error loading prior equation from {eq_path}: {e}")
                    prior_equations.append("ERROR")
        
        # Load equations from all other models
        model_equations = {}
        for model_name, exp_path in models.items():
            memory_slots_dir = os.path.join(exp_path, 'logs/experiment_metrics/equations/memory_slots')
            
            if not os.path.exists(memory_slots_dir):
                continue
            
            memory_slots = sorted([d for d in os.listdir(memory_slots_dir) if d.startswith('memory_slot_')])
            
            equations = []
            for mem_slot in memory_slots:
                mem_slot_dir = os.path.join(memory_slots_dir, mem_slot)
                eq_files = sorted([f for f in os.listdir(mem_slot_dir) if f.startswith('equation_') and f.endswith('.pkl')])
                
                for eq_file in eq_files:
                    eq_path = os.path.join(mem_slot_dir, eq_file)
                    try:
                        with open(eq_path, 'rb') as f:
                            eq = dill.load(f)
                        equations.append(str(eq))
                    except Exception as e:
                        print(f"Error loading equation from {eq_path}: {e}")
                        equations.append("ERROR")
            
            model_equations[model_name] = equations
        
        # Create rows - one per equation
        n_equations = len(prior_equations)
        for eq_idx in range(n_equations):
            row = {
                'dataset': dataset_name,
                'seed': seed,
                'equation_idx': eq_idx,
                'prior_equation': prior_equations[eq_idx] if eq_idx < len(prior_equations) else "N/A"
            }
            
            # Add equations from each model
            for model_name, equations in model_equations.items():
                if model_name == 'prior_symbolic_cbm':
                    continue  # Already added
                row[f'{model_name}_equation'] = equations[eq_idx] if eq_idx < len(equations) else "N/A"
            
            results.append(row)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns: dataset, seed, equation_idx, prior_equation, then other models
    if len(results_df) > 0:
        fixed_cols = ['dataset', 'seed', 'equation_idx', 'prior_equation']
        model_cols = sorted([col for col in results_df.columns if col.endswith('_equation') and col != 'prior_equation'])
        results_df = results_df[fixed_cols + model_cols]
    
    return results_df

# def plot_licem_weights_distribution(train_w, test_w, test_w_lcmr, result_figs, c_names, y_names, title_font=None, label_font=None, tick_font=None, legend_font=None):
#     """
#     Plot the LICEM weights distribution and the m-cbm lin weights distribution.
#     """

#     n_concepts = len(c_names)
#     n_tasks = len(y_names)

#     cnt = 0

#     for i in tqdm(range(n_concepts)):
#         for j in range(n_tasks):
#             licem_weights = test_w['00'][:, i, j].flatten().numpy()
#             lcmr_weights = test_w_lcmr.squeeze()[:, j, i].cpu().numpy()

#             fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            
#             # Plot LICEM weights as smooth histogram (continuous distribution)
#             counts, bins = np.histogram(licem_weights, bins=50, density=True)
#             bin_centers = (bins[:-1] + bins[1:]) / 2
#             # Convert density to probability by multiplying by bin width
#             bin_width = bins[1] - bins[0]
#             probabilities_licem = counts * bin_width
#             ax.plot(bin_centers, probabilities_licem, color='blue', linewidth=2, alpha=0.8)
#             ax.fill_between(bin_centers, probabilities_licem, alpha=0.3, color='blue')

#             # Plot M-CBM-lin weights as bar plot (discrete distribution)
#             unique_values, counts = np.unique(lcmr_weights, return_counts=True)
#             probabilities_lcmr = counts / len(lcmr_weights)
#             ax.bar(unique_values, probabilities_lcmr, alpha=0.7, color='green', width=0.1)
            
#             ax.set_xlabel('Weights', fontdict=label_font)
#             ax.set_ylabel('Probability', fontdict=label_font)
#             ax.grid(True, alpha=0.3)
#             ax.minorticks_off()
#             if tick_font:   
#                 ax.tick_params(axis='both', which='major', labelsize=tick_font.get('size', 12))

#             plt.tight_layout()
#             # Save the combined plot
#             path = os.path.join(result_figs, 'licem_vs_lcmr_weights_distribution')
#             os.makedirs(path, exist_ok=True)
#             plt.savefig(f"{path}/{c_names[i]}_{y_names[j]}.pdf", bbox_inches='tight')
#             plt.close()

#             if cnt>10:
#                 return
#             cnt += 1
