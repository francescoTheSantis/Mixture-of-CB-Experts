import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import torch
import pandas as pd
import scienceplots

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
    n_cols = len(unique_noises)
    n_rows = len(unique_datasets)
    fig, axes = plt.subplots(n_cols, n_rows, figsize=(30, 7), sharex=True, sharey=False)
    
    for i, dataset in enumerate(unique_datasets):
        for j, noise in enumerate(unique_noises):
            ax = axes[i] #axes[i, j] if n_rows > 1 else axes[j]
            data = df[(df['noise'] == noise) & (df['dataset'] == dataset)]
            if dataset in ['cebab']:
                metric = 'mse'
            else:
                metric = 'accuracy'
            grouped_data = data.groupby(['p_int', 'model']).agg(
                mean_metric=(metric, 'mean'),
                std_metric=(metric, 'std')
            ).reset_index().fillna(0)
            
            if relative_accuracy:
                # Calculate relative accuracy for each model
                for model in grouped_data['model'].unique():
                    model_data = grouped_data[grouped_data['model'] == model]
                    baseline = model_data[model_data['p_int'] == 0]['mean_metric'].iloc[0] if len(model_data[model_data['p_int'] == 0]) > 0 else 0
                    grouped_data.loc[grouped_data['model'] == model, 'mean_metric'] -= baseline
            
            for model in grouped_data['model'].unique():
                model_data = grouped_data[grouped_data['model'] == model]
                style = model_styles.get(model, {'marker': 'o', 'color': 'black', 'size': 10, 'name': model})
                ax.plot(model_data['p_int'], model_data['mean_metric'], color=style['color'], linestyle='-', alpha=0.5)
                ax.scatter(model_data['p_int'], model_data['mean_metric'], marker=style['marker'], color=style['color'], s=style['size']**2, label=style['name'], edgecolor='black', alpha=0.5)
                ax.fill_between(model_data['p_int'], model_data['mean_metric'] - model_data['std_metric'], model_data['mean_metric'] + model_data['std_metric'], color=style['color'], alpha=0.2)
            ax.set_xlabel('$p_{int}$', fontsize=label_font['size'])
            if j == 0:
                ax.set_title(f'{get_df_name(dataset)}', fontsize=label_font['size'])
            if dataset not in ['cebab']:
                ylabel = 'Relative Task Acc' if relative_accuracy else 'Task Acc'
                ax.set_ylabel(ylabel, fontsize=label_font['size'])
            else:
                ylabel = 'Relative Task MSE' if relative_accuracy else 'Task MSE'
                ax.set_ylabel(ylabel, fontsize=label_font['size'])
            ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
            ax.minorticks_off()
            ax.grid(True)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))
    
    # Create a single legend below the plots
    handles, labels = [], []
    for ax in axes.flatten():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    for handle in handles:
        handle.set_alpha(1)  # Remove transparency from legend markers

    # Create custom legend handles
    custom_handles = [plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=style['size']+10, label=style['name'], markeredgewidth=0.5, markeredgecolor='black') for style in model_styles.values()]

    # Create a single legend below the plots
    fig.legend(
        handles=custom_handles,
        loc='lower center',
        ncol=(len(custom_handles) + 1) // 2,  # Split legend into two rows
        fontsize=tick_font['size'],
        frameon=True,
        bbox_to_anchor=(0.5, -0.2),
        columnspacing=1.0,
        handletextpad=0.5
    )

    plt.tight_layout()
    str_store = str(unique_noises[0]).replace('.', '')
    suffix = '_relative' if relative_accuracy else ''
    plt.savefig(f'figs/intervention_{str_store}{suffix}.pdf')
    plt.show()


def plot_memory_ablation(performance, model_styles, title_font, label_font, tick_font):

    # Sort the points in the order of memory_size.
    performance = performance.sort_values(by=['dataset', 'memory_size'])

    num_seeds = performance['seed'].nunique()

    # Avg over the seeds for the performance metrics
    performance = performance.groupby(['dataset', 'memory_size', 'model']).agg(
        mean_task=('task', 'mean'),
        std_task=('task', 'std'),
        mean_concept=('concept', 'mean'),
        std_concept=('concept', 'std')
    ).reset_index()

    
    # instead of the std compute the standard error at 95% confidence
    performance['se_task'] = 1.96 * performance['std_task'] / np.sqrt(num_seeds)
    performance['se_concept'] = 1.96 * performance['std_concept'] / np.sqrt(num_seeds)

    fig, axes = plt.subplots(1, len(performance['dataset'].unique()), figsize=(25, 10), sharey=False)

    for idx, dataset in enumerate(performance['dataset'].unique()):
        ax = axes[idx] if len(performance['dataset'].unique()) > 1 else axes
        data = performance[performance['dataset'] == dataset]
        for model in data['model'].unique():
            model_data = data[data['model'] == model]
            ax.plot(
                model_data['memory_size'], 
                model_data['mean_task'], 
                label=model_styles[model]['name'], 
                marker=model_styles[model]['marker'], 
                color=model_styles[model]['color'], 
                markersize=model_styles[model]['size']
            )
            ax.fill_between(
                model_data['memory_size'], 
                model_data['mean_task'] - model_data['se_task'], 
                model_data['mean_task'] + model_data['se_task'], 
                color=model_styles[model]['color'], 
                alpha=0.2
            )
        ax.set_title(get_df_name(dataset), fontdict=title_font)
        ax.set_xlabel('Memory Size', fontdict=label_font)
        ax.set_ylabel('Task Accuracy', fontdict=label_font)
        ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
        ax.minorticks_off()
        ax.grid(True, zorder=0)

    # Filter the style according to the models' names which are present in the performance df
    filtered_styles = {name: style for name, style in model_styles.items() if name in performance['model'].values}

    # Create custom legend handles
    custom_handles = [plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=(style['size']-3), label=style['name'], markeredgewidth=0.5, markeredgecolor='black') for style in filtered_styles.values()]

    # Create a single legend below the plots
    fig.legend(handles=custom_handles, loc='lower center', ncol=(len(custom_handles) + 1) // 2, fontsize=tick_font['size'], frameon=True, bbox_to_anchor=(0.5, -0.15), columnspacing=1.0, handletextpad=0.5)

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