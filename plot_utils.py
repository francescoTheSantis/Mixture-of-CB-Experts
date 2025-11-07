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
        return 'dSprites-Memory'
    elif df=='mnist_arithmetic':
        return 'MNIST-Arith.'
    elif df=='pendulum':
        return 'Pendulum'
    elif df=='mawps':
        return 'MAWPS'

def get_exp_from_path(paths):
    # Collect all the experiments in the given paths
    exps_path = []
    lmr_paths = []
    for path in paths:
        exps = os.listdir(path)
        exps_path += [os.path.join(path, exp) for exp in exps if 'multirun' not in exp]

    performance = pd.DataFrame()

    # Iterate over all the experiments and collect the performance metrics and the config
    for exp in exps_path:
        d = {}
        conf_file = os.path.join(exp, '.hydra/config.yaml')
        result_file = os.path.join(exp, 'logs/experiment_metrics/version_0/metrics.csv') 
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
                    result = pd.read_csv(file, header=0)

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

                print(d)
                
                if d['model'] == 'l_cmr' and d['seed']==1:
                    expl_dict = d.copy()
                    expl_dict['path'] = exp
                    lmr_paths.append(expl_dict)

                performance = pd.concat([performance, pd.DataFrame([d])], ignore_index=True)
        except Exception as e:
            print(f"Error while processing {exp}: {e}")
            continue

    # Filter the sym_cmr_prior according to the combination of memory size and dataset
    # if dataset==dsprites_simple, keep only memory_size==1
    # if dataset==mnist_arithmetic, keep only memory_size==4
    # if dataset==dsprites_complex, keep only memory_size==3
    # if dataset==pendulum, keep only memory_size==1
    condition = (
        ((performance['model'] == 'm_sym_cmr_prior') & (performance['dataset'] == 'dsprites_simple') & (performance['memory_size'] == 1)) |
        ((performance['model'] == 'm_sym_cmr_prior') & (performance['dataset'] == 'mnist_arithmetic') & (performance['memory_size'] == 4)) |
        ((performance['model'] == 'm_sym_cmr_prior') & (performance['dataset'] == 'dsprites_complex') & (performance['memory_size'] == 3)) |
        ((performance['model'] == 'm_sym_cmr_prior') & (performance['dataset'] == 'pendulum') & (performance['memory_size'] == 1)) |
        (performance['model'] != 'm_sym_cmr_prior')
    )
    performance = performance[condition]

    return performance, lmr_paths

def get_intervention_from_path(paths, filtered_exps=None):
    performance = pd.DataFrame()

    exps_path = []
    lmr_paths = []
    for path in paths:
        exps = os.listdir(path)
        exps_path += [os.path.join(path, exp) for exp in exps if 'multirun' not in exp]

    for exp in exps_path:
        conf_file = os.path.join(exp, '.hydra/config.yaml')
        result_file = os.path.join(exp, 'logs/experiment_metrics/version_0/interventions.csv')        
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

def plot_intervention_results(df, 
                                metric='accuracy', 
                                unique_noises=[0.0], 
                                title_font=None, 
                                label_font=None, 
                                tick_font=None, 
                                legend_font=None,
                                custom_order=None,
                                model_styles=None,
                                relative_accuracy=True):
    unique_datasets = custom_order
    
    # NOTE: keep in mind to update this list if you add new regression datasets
    regression_datasets = ['mnist_arithmetic', 'dsprites_simple', 'dsprites_complex', 'cebab', 'pendulum', 'mawps']
    
    # Separate datasets by task type
    classification_datasets = [d for d in unique_datasets if d not in regression_datasets]
    regression_datasets = [d for d in unique_datasets if d in regression_datasets]
    
    # Organize datasets with classification first, then regression
    organized_datasets = classification_datasets + regression_datasets
    
    n_datasets = len(organized_datasets)
    n_cols = max(len(classification_datasets), len(regression_datasets))
    n_rows = 2  # Force 2 rows: classification on first row, regression on second

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
        if dataset in classification_datasets:
            row = 0
            col = classification_datasets.index(dataset)
        else:
            row = 1
            col = regression_datasets.index(dataset)
        
        if n_rows == 1:
            ax = axes[col] if n_cols > 1 else axes[0]
        else:
            ax = axes[row][col] if n_cols > 1 else axes[row][0] if col == 0 else axes[row][col]
        
        for noise in unique_noises:
            data = df[(df['noise'] == noise) & (df['dataset'] == dataset)]
            if dataset in regression_datasets:
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
            if dataset not in regression_datasets:
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
        if row == 1:
            ax.set_xlabel('$p_{int}$', fontsize=label_font['size'])
        
        ax.set_title(f'{get_df_name(dataset)}', fontsize=label_font['size'])

        # Set ylabel only for the leftmost subplot in each row
        if col == 0:
            if dataset not in regression_datasets:
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
    total_datasets = len(classification_datasets) + len(regression_datasets)
    legend_in_subplot = False
    
    # Hide unused subplots in first row
    for col in range(len(classification_datasets), n_cols):
        axes[0][col].set_visible(False)
    
    # Hide unused subplots in second row and check for legend placement
    for col in range(len(regression_datasets), n_cols):
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
    os.makedirs(f'figs/intervention/{suffix}', exist_ok=True)
    plt.savefig(f'figs/intervention/{suffix}/{str_store}.pdf')
    plt.show()

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
    plt.savefig('figs/memory_ablation.pdf')

    plt.show()


def plot_explanations(lmr_paths):

    for exp in lmr_paths:
        try:
            # If it does not exist, create the figs directory
            figs_path = f'figs/{exp["model"]}_explanations/{exp["dataset"]}'
            if not os.path.exists(figs_path):
                os.makedirs(figs_path)
            exp_info_path = os.path.join(exp['path'], 'logs/experiment_metrics/version_0')
            # Load all the required files
            pred_CBMs = torch.load(os.path.join(exp_info_path, 'pred_CBMs.pt'))
            c_trues = pd.read_csv(os.path.join(exp_info_path, 'c_trues.csv'))
            c_preds = pd.read_csv(os.path.join(exp_info_path, 'c_preds.csv'))
            y_trues = pd.read_csv(os.path.join(exp_info_path, 'y_trues.csv'))
            y_preds = pd.read_csv(os.path.join(exp_info_path, 'y_preds.csv'))
            binary_classification = True if len(y_trues.columns) == 2 else False

            # Sample N random samples from the test-set
            n_samples = 20
            random_indices = np.random.choice(len(y_trues), n_samples, replace=False)

            # Plot the explanations
            for i in range(n_samples):
                idx = random_indices[i]
                # Get the true and predicted concepts
                c_true = c_trues.iloc[idx].values
                c_pred = (c_preds.iloc[idx].values > 0.5).astype(int)

                # get the column name of the concepts corresponding to the highest value
                c_name_pred = [x.replace('_',' ') for idx, x in enumerate(c_preds.columns) if idx in np.argwhere(c_pred>0.5)]
                c_name_true = [x.replace('_',' ') for idx, x in enumerate(c_trues.columns) if idx in np.argwhere(c_true>0.5)]
                # Get the true and predicted labels
                if binary_classification:
                    y_true = y_trues.iloc[idx,1]
                    y_pred = y_preds.iloc[idx,1]
                else:
                    y_true = y_trues.iloc[idx].values.argmax(-1)
                    y_pred = y_preds.iloc[idx].values.argmax(-1)
                # get the column name of the concepts corresponding to the highest value
                y_name_pred = y_preds.columns[y_pred].replace('_',' ')
                y_name_true = y_trues.columns[y_true].replace('_',' ')
                # Get the weights associated to the predicted class of the 
                # predicted CBM
                if binary_classification:
                    y_pred = 0
                pred_CBM = pred_CBMs[idx,y_pred,:,:].squeeze().cpu().numpy()
                # Perform the element-wise multiplication
                logits = np.multiply(pred_CBM, c_pred)

                # Sort the logits by absolute value in ascending order and take the k highest values
                top_k = 10
                indices = np.argsort(np.abs(logits))[::-1][:top_k]
                logits = logits[indices]
                c_preds_names = [c_preds.columns[idx] for idx in indices]
                # c_trues_names = [c_trues.columns[idx] for idx in indices]

                # Plot the explanations
                fig, ax = plt.subplots(figsize=(10, 5))
                bar_colors = ['tab:blue' if val >= 0 else 'tab:red' for val in logits]
                y_pos = np.arange(len(logits))
                ax.barh(
                    y_pos, logits, color=bar_colors,
                    edgecolor='black', linewidth=1.5, alpha=0.6
                )
                # Add black vertical line at 0
                ax.axvline(x=0, color='black', linewidth=0.9)
                ax.set_yticks(y_pos)
                ax.set_yticklabels([x.replace('_',' ') for x in c_preds_names], 
                                    fontsize=14)
                ax.set_xlabel('Logit Value', fontsize=12)
                ax.set_title(f'Predicted class: {y_name_pred}, True class: {y_name_true}', 
                            fontsize=12)
                # Eliminate minor ticks
                ax.xaxis.set_minor_locator(plt.NullLocator())
                ax.yaxis.set_minor_locator(plt.NullLocator())
                # Save the figure
                plt.tight_layout()
                plt.savefig(f'{figs_path}/explanations_{i}.pdf')
        except:
            print(f"Error while plotting explanations for {exp['model']} on {exp['dataset']}. Skipping...")
            continue
    print("Explanations plotted successfully.")


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
    plt.savefig(f'figs/concept_size_ablation.pdf', bbox_inches='tight')


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



def plot_licem_weights_distribution(train_w, test_w, test_w_lcmr, result_figs, c_names, y_names, title_font=None, label_font=None, tick_font=None, legend_font=None):
    """
    Plot the LICEM weights distribution and the m-cbm lin weights distribution.
    """

    n_concepts = len(c_names)
    n_tasks = len(y_names)

    cnt = 0

    for i in tqdm(range(n_concepts)):
        for j in range(n_tasks):
            licem_weights = test_w['00'][:, i, j].flatten().numpy()
            lcmr_weights = test_w_lcmr.squeeze()[:, j, i].cpu().numpy()

            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            
            # Plot LICEM weights as smooth histogram (continuous distribution)
            counts, bins = np.histogram(licem_weights, bins=50, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            # Convert density to probability by multiplying by bin width
            bin_width = bins[1] - bins[0]
            probabilities_licem = counts * bin_width
            ax.plot(bin_centers, probabilities_licem, color='blue', linewidth=2, alpha=0.8)
            ax.fill_between(bin_centers, probabilities_licem, alpha=0.3, color='blue')

            # Plot M-CBM-lin weights as bar plot (discrete distribution)
            unique_values, counts = np.unique(lcmr_weights, return_counts=True)
            probabilities_lcmr = counts / len(lcmr_weights)
            ax.bar(unique_values, probabilities_lcmr, alpha=0.7, color='green', width=0.1)
            
            ax.set_xlabel('Weights', fontdict=label_font)
            ax.set_ylabel('Probability', fontdict=label_font)
            ax.grid(True, alpha=0.3)
            ax.minorticks_off()
            if tick_font:   
                ax.tick_params(axis='both', which='major', labelsize=tick_font.get('size', 12))

            plt.tight_layout()
            # Save the combined plot
            path = os.path.join(result_figs, 'licem_vs_lcmr_weights_distribution')
            os.makedirs(path, exist_ok=True)
            plt.savefig(f"{path}/{c_names[i]}_{y_names[j]}.pdf", bbox_inches='tight')
            plt.close()

            if cnt>10:
                return
            cnt += 1

def plot_pareto_front(performance, model_styles, title_font, label_font, tick_font, custom_order):

    performance = compute_avg_and_uncertainty(performance, custom_order)

    # save the performance as csv
    performance.to_csv('tabs/memory_ablation_performance.csv', index=False)

    # Define operational complexity for each model
    oc = { 
        'licem': 2,
        'dcr': 3,
        'cmr': 3,
        'l_cmr': 2,
        'm_sym_cmr_kan': 7,
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
                    # Use cbm_linear style if l_cmr has memory_size = 1
                    if model == 'l_cmr' and memory_size == 1:
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

    # Create filtered styles including both original models and cbm_linear for m_sym_cmr_kan with memory_size=1
    filtered_styles = {}
    for name, style in model_styles.items():
        if name in performance['model'].values:
            filtered_styles[name] = style
    
    # Add cbm_linear style if m_sym_cmr_kan appears with memory_size=1
    if 'm_sym_cmr_kan' in performance['model'].values and 'cbm_linear' in model_styles:
        if any((performance['model'] == 'm_sym_cmr_kan') & (performance['memory_size'] == 1)):
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
    plt.savefig('figs/pareto_front.pdf')


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
        'l_cmr': 2,
        'm_sym_cmr_kan': 7,
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
                    
                    # Use cbm_linear style if m_sym_cmr_kan has memory_size = 1
                    if model == 'm_sym_cmr_kan' and memory_size == 1:
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
    
    # Create filtered styles including both original models and cbm_linear for m_sym_cmr_kan with memory_size=1
    filtered_styles = {}
    for name, style in model_styles.items():
        if name in df['model'].values:
            filtered_styles[name] = style

    # Add cbm_linear style if l_cmr appears with memory_size=1
    if 'l_cmr' in df['model'].values and 'cbm_linear' in model_styles:
        if any((df['model'] == 'l_cmr') & (df['memory_size'] == 1)):
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
    os.makedirs(f'figs/intervention_memory_pareto/{suffix}', exist_ok=True)
    plt.savefig(f'figs/intervention_memory_pareto/{suffix}/interventions_pint_{str_store}.pdf')
    plt.show()

