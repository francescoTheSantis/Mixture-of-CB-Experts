import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import warnings
import os
import yaml
from matplotlib.ticker import FuncFormatter
from src.utilities import plot_explanations

# I used scienceplots for the style of the plots, but you can use any other style you want.
warnings.filterwarnings("ignore")
plt.style.use(['science', 'ieee', 'no-latex'])

# List the paths containing the results
paths = [
    
]

###### Collect results regarding concept/task performance ######

exps_path = []
lmr_paths = []
for path in paths:
    exps = os.listdir(path)
    exps_path += [os.path.join(path, exp) for exp in exps if 'multirun' not in exp]

performance = pd.DataFrame()

for exp in exps_path:
    d = {}
    conf_file = os.path.join(exp, '.hydra/config.yaml')
    result_file = os.path.join(exp, 'logs/experiment_metrics/version_0/metrics.csv')  
    if os.path.exists(conf_file) and os.path.exists(result_file):
        try:
            with open(conf_file, 'r') as file:
                conf = yaml.safe_load(file)
            d['seed'] = conf['seed']
            d['dataset'] = conf['dataset']['metadata']['name']
            d['model'] = conf['model']['metadata']['name']

            with open(result_file, 'r') as file:
                result = pd.read_csv(file, header=0)

            # Select the last row of the dataframe where we test the model
            d['task'] = result['test_task_acc'].iloc[-1]
            if conf['model']['metadata']['name']=='blackbox':
                d['concept'] = 0
            else:
                d['concept'] = result['test_concept_acc'].iloc[-1]

            print(d)
            
            if d['model'] == 'm_licem' and d['seed']==1:
                expl_dict = d.copy()
                expl_dict['path'] = exp
                lmr_paths.append(expl_dict)

            performance = pd.concat([performance, pd.DataFrame([d])], ignore_index=True)
        except:
            pass


######### Dataset and model styles #########

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

marker_size = 14

# Define a dictionary to associate marker, name, and color to each model.
# If the experiment you run does not contain a model, just remove it from the dictionary.
# If you want to add a new model, just add it to the dictionary.
model_styles = {
    'cem': {'marker': 'P', 'name': 'CEM', 'color': 'tab:orange', 'size': marker_size},
    'cbm_linear': {'marker': '*', 'name': 'CBM+Linear', 'color': 'tab:olive', 'size': marker_size},
    'cbm_mlp': {'marker': '^', 'name': 'CBM+MLP', 'color': 'tab:red', 'size': marker_size},
    'blackbox': {'marker': 'o', 'name': 'BlackBox', 'color': 'tab:purple', 'size': marker_size},
    'cmr': {'marker': 'v', 'name': 'CMR', 'color': 'tab:pink', 'size': marker_size},
    'dcr': {'marker': 'h', 'name': 'DCR', 'color': 'tab:gray', 'size': marker_size},
    'm_licem': {'marker': 's', 'name': 'M-LICEM', 'color': 'tab:blue', 'size': marker_size},
    #'m_licem_sampling': {'marker': 'D', 'name': 'M-LICEM', 'color': 'tab:cyan', 'size': marker_size},
    #'m_licem_int_sel_sampling': {'marker': 'X', 'name': 'M-LICEM+IntSel+Sampling', 'color': 'tab:green', 'size': marker_size},
    #'m_licem_neg_weights': {'marker': '<', 'name': 'M-LICEM+NegWeights', 'color': 'tab:brown', 'size': marker_size},
    #'m_licem_neg_weights_sampling': {'marker': '>', 'name': 'M-LICEM+NegWeights+Sampling', 'color': 'tab:olive', 'size': marker_size},
    #'m_licem_neg_weights_int_sel_sampling': {'marker': '8', 'name': 'M-LICEM+NegWeights+IntSel+Sampling', 'color': 'tab:gray', 'size': marker_size},
    #'m_licem_proto': {'marker': 'D', 'name': 'ProtoCBM', 'color': 'tab:green', 'size': marker_size},
    #'m_licem_proto_int_sel': {'marker': 's', 'name': 'ProtoCBM+IntSel', 'color': 'tab:blue', 'size': marker_size},
}

# Define the custom order
# If the experiment you run does not contain a dataset, just remove it from the list.
custom_order = ['xor', \
                #'dot', \
                #'checkmark', \
                #'trigonometry', \
                'mnist_addition', \
                'cub', \
                'awa2',
                'awa2_incomplete',
                'cub_incomplete'
                ]


# Filter the performance dataframe to keep only the models in model_styles 
# and datasets in custom_order.
performance = performance[performance['model'].isin(model_styles.keys()) & \
                          performance['dataset'].isin(custom_order)]


######### Only for the LinearMemoryReasoner with seed=1, plot explanations #########
plot_explanations(lmr_paths)


########## Task & Concept Accuracy Plot ##########

#df = performance.copy()
# Filter data for 'task' and 'concept'
task_df = performance.copy()
task_df = task_df.rename(columns={'task': 'accuracy'})
concept_df = performance.copy()
concept_df = concept_df.rename(columns={'concept': 'accuracy'})

# Compute mean and std for 'task'
task_stats = task_df.groupby(['model', 'dataset']).agg(
    avg_accuracy_task=('accuracy', 'mean'),
    std_accuracy_task=('accuracy', 'std')
).reset_index().fillna(0)

# Compute mean and std for 'concept'
concept_stats = concept_df.groupby(['model', 'dataset']).agg(
    avg_accuracy_concept=('accuracy', 'mean'),
    std_accuracy_concept=('accuracy', 'std')
).reset_index().fillna(0)

# Merge the two DataFrames on 'model' and 'dataset'
merged_stats = pd.merge(task_stats, concept_stats, on=['model', 'dataset'])

# Define font properties
title_font = {'size': 24, 'weight': 'bold'}
label_font = {'size': 24}
tick_font = {'size': 10}

merged_stats = merged_stats.sort_values('dataset')
merged_stats['dataset'] = pd.Categorical(merged_stats['dataset'], categories=custom_order, ordered=True)

fig, axes = plt.subplots(1, len(merged_stats['dataset'].unique()), figsize=(15, 4), sharey=False, sharex=True)

for idx, dataset in enumerate(merged_stats['dataset'].unique()):
    if len(merged_stats['dataset'].unique()) == 1:
        ax = axes
    else:
        ax = axes[idx]
    data = merged_stats[merged_stats['dataset'] == dataset]
    for model in data['model']:
        model_data = data[data['model'] == model]
        style = model_styles[model]
        ax.errorbar(model_data['avg_accuracy_concept'], model_data['avg_accuracy_task'],
                    xerr=model_data['std_accuracy_concept'], yerr=model_data['std_accuracy_task'],
                    fmt=style['marker'], label=style['name'], color=style['color'], markersize=style['size'],
                    markeredgewidth=0.5, markeredgecolor='black', alpha=0.7)
    ax.set_title(get_df_name(dataset), fontdict=title_font)
    ax.tick_params(axis='both', which='major', labelsize=tick_font['size'])
    ax.minorticks_off()
    ax.grid(True, zorder=0)
    if idx == 0:
        ax.set_ylabel('Task Acc', fontdict=label_font)
    ax.set_xlabel('Concept Acc', fontdict=label_font)

# Create custom legend handles
custom_handles = [plt.Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor=style['color'], markersize=(style['size']-3), label=style['name'], markeredgewidth=0.5, markeredgecolor='black') for style in model_styles.values()]

# Create a single legend below the plots
fig.legend(handles=custom_handles, loc='lower center', ncol=(len(custom_handles) + 1) // 2, fontsize=tick_font['size'], frameon=True, bbox_to_anchor=(0.5, -0.15), columnspacing=1.0, handletextpad=0.5)

plt.tight_layout()
plt.savefig('figs/performance.pdf')

plt.show()



########## Task Accuracy Table ##########

task_avg = task_stats[['model', 'dataset', 'avg_accuracy_task']]
task_std = task_stats[['model', 'dataset', 'std_accuracy_task']]

# Merge task_avg and task_std dataframes on 'model' and 'dataset'
merged_task = pd.merge(task_avg, task_std, on=['model', 'dataset'])

# Create a pivot table with the desired format
pivot_table_avg = task_avg.pivot(index='model', columns='dataset', values=['avg_accuracy_task'])
pivot_table_avg.columns = pivot_table_avg.columns.get_level_values(1)
pivot_table_std = task_std.pivot(index='model', columns='dataset', values=['std_accuracy_task'])
pivot_table_std.columns = pivot_table_std.columns.get_level_values(1)

final_table = pd.DataFrame()
for i, row in pivot_table_avg.iterrows():
    d={}
    for j in pivot_table_std.columns:
        acc = row[j]*100
        std = pivot_table_std.loc[i, j]*100
        d[j] = f"{acc:.2f} ± {std:.2f}"
    # add a column to the final_table dataframe called row.name which contains d
    final_table = pd.concat([final_table, pd.DataFrame(d, index=[row.name])], axis=0)
    
# Reindex the columns of final_table according to the custom order
final_table = final_table.reindex(columns=custom_order)

# Replace the model and dataset names
final_table.index = final_table.index.map(lambda x: model_styles[x]['name'] if x in model_styles else x)
final_table.columns = final_table.columns.map(lambda x: get_df_name(x))

print('\n\nTask Accuracy Table:')
print('-------------------')
print(final_table)

# store the table in a csv file
final_table.to_csv('figs/task_accuracy.csv', index=True)


########## Concept Accuracy Table ##########

task_avg = concept_stats[['model', 'dataset', 'avg_accuracy_concept']]
task_std = concept_stats[['model', 'dataset', 'std_accuracy_concept']]

# Merge task_avg and task_std dataframes on 'model' and 'dataset'
merged_task = pd.merge(task_avg, task_std, on=['model', 'dataset'])

# Create a pivot table with the desired format
pivot_table_avg = task_avg.pivot(index='model', columns='dataset', values=['avg_accuracy_concept'])
pivot_table_avg.columns = pivot_table_avg.columns.get_level_values(1)
pivot_table_std = task_std.pivot(index='model', columns='dataset', values=['std_accuracy_concept'])
pivot_table_std.columns = pivot_table_std.columns.get_level_values(1)

final_table = pd.DataFrame()
for i, row in pivot_table_avg.iterrows():
    d={}
    for j in pivot_table_std.columns:
        acc = row[j]*100
        std = pivot_table_std.loc[i, j]*100
        d[j] = f"{acc:.2f} ± {std:.2f}"
    # add a column to the final_table dataframe called row.name which contains d
    final_table = pd.concat([final_table, pd.DataFrame(d, index=[row.name])], axis=0)
    
# Reindex the columns of final_table according to the custom order
final_table = final_table.reindex(columns=custom_order)

# Replace the model and dataset names
final_table.index = final_table.index.map(lambda x: model_styles[x]['name'] if x in model_styles else x)
final_table.columns = final_table.columns.map(lambda x: get_df_name(x))

print('\n\nConcept Accuracy Table:')
print('-------------------')
print(final_table)

# store the table in a csv file
final_table.to_csv('figs/concept_accuracy.csv', index=True)


########## Collect intervention results ##########

performance = pd.DataFrame()

for exp in exps_path:
    conf_file = os.path.join(exp, '.hydra/config.yaml')
    result_file = os.path.join(exp, 'logs/experiment_metrics/version_0/interventions.csv')        
    if os.path.exists(conf_file) and os.path.exists(result_file):
        with open(result_file, 'r') as file:
            d = pd.read_csv(result_file)[['noise','p_int','f1','accuracy']]
        
        with open(conf_file, 'r') as file:
            conf = yaml.safe_load(file)
        d['seed'] = conf['seed']
        d['dataset'] = conf['dataset']['metadata']['name']
        d['model'] = conf['model']['metadata']['name']

        performance = pd.concat([performance, d], ignore_index=True)

# Filter the performance dataframe to keep only the models in model_styles 
# and datasets in custom_order.
performance = performance[performance['model'].isin(model_styles.keys()) & \
                          performance['dataset'].isin(custom_order)]


########## Intervention plots ########## 

def plot_intervention_results(df, metric='accuracy', title_font=None, label_font=None, tick_font=None, legend_font=None):
    unique_noises = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
    unique_datasets = custom_order
    n_cols = len(unique_noises)
    n_rows = len(unique_datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey='row')
    
    for i, dataset in enumerate(unique_datasets):
        for j, noise in enumerate(unique_noises):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            data = df[(df['noise'] == noise) & (df['dataset'] == dataset)]
            grouped_data = data.groupby(['p_int', 'model']).agg(
                mean_metric=(metric, 'mean'),
                std_metric=(metric, 'std')
            ).reset_index().fillna(0)
            for model in grouped_data['model'].unique():
                model_data = grouped_data[grouped_data['model'] == model]
                style = model_styles.get(model, {'marker': 'o', 'color': 'black', 'size': 10, 'name': model})
                ax.plot(model_data['p_int'], model_data['mean_metric'], color=style['color'], linestyle='-', alpha=0.5)
                ax.scatter(model_data['p_int'], model_data['mean_metric'], marker=style['marker'], color=style['color'], s=style['size']**2, label=style['name'], edgecolor='black', alpha=0.5)
                ax.fill_between(model_data['p_int'], model_data['mean_metric'] - model_data['std_metric'], model_data['mean_metric'] + model_data['std_metric'], color=style['color'], alpha=0.2)
            if i == 0:
                ax.set_title(r'$\theta$'+f'={noise}', fontsize=title_font['size'])
            if i == n_rows - 1:
                ax.set_xlabel('$p_{int}$', fontsize=label_font['size'])
            if j == 0:
                ax.set_ylabel(f'{get_df_name(dataset)}', fontsize=label_font['size'])
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
    fig.legend(handles=custom_handles, loc='lower center', ncol=len(custom_handles), fontsize=tick_font['size'], frameon=True, bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    plt.savefig('figs/intervention.pdf')
    plt.show()

# Call the function with the desired metric and font properties
legend_font = {'size': 44}
title_font = {'size': 44, 'weight': 'bold'}
label_font = {'size': 44}
tick_font = {'size': 30}
plot_intervention_results(performance, metric='accuracy', title_font=title_font, label_font=label_font, tick_font=tick_font, legend_font=legend_font)



########## Intervention ID plots ##########

def plot_intervention_results(df, metric='accuracy', title_font=None, label_font=None, tick_font=None, legend_font=None):
    unique_noises = [0]
    unique_datasets = custom_order
    n_cols = len(unique_noises)
    n_rows = len(unique_datasets)
    fig, axes = plt.subplots(n_cols, n_rows, figsize=(30, 7), sharex=True, sharey=False)
    
    for i, dataset in enumerate(unique_datasets):
        for j, noise in enumerate(unique_noises):
            ax = axes[i] #axes[i, j] if n_rows > 1 else axes[j]
            data = df[(df['noise'] == noise) & (df['dataset'] == dataset)]
            grouped_data = data.groupby(['p_int', 'model']).agg(
                mean_metric=(metric, 'mean'),
                std_metric=(metric, 'std')
            ).reset_index().fillna(0)
            for model in grouped_data['model'].unique():
                model_data = grouped_data[grouped_data['model'] == model]
                style = model_styles.get(model, {'marker': 'o', 'color': 'black', 'size': 10, 'name': model})
                ax.plot(model_data['p_int'], model_data['mean_metric'], color=style['color'], linestyle='-', alpha=0.5)
                ax.scatter(model_data['p_int'], model_data['mean_metric'], marker=style['marker'], color=style['color'], s=style['size']**2, label=style['name'], edgecolor='black', alpha=0.5)
                ax.fill_between(model_data['p_int'], model_data['mean_metric'] - model_data['std_metric'], model_data['mean_metric'] + model_data['std_metric'], color=style['color'], alpha=0.2)
            ax.set_xlabel('$p_{int}$', fontsize=label_font['size'])
            if j == 0:
                ax.set_title(f'{get_df_name(dataset)}', fontsize=label_font['size'])
            if i == 0:
                ax.set_ylabel('Task Acc', fontsize=label_font['size'])
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
    plt.savefig('figs/intervention_id.pdf')
    plt.show()

# Call the function with the desired metric and font properties
legend_font = {'size': 44}
title_font = {'size': 36, 'weight': 'bold'}
label_font = {'size': 44}
tick_font = {'size': 28}
plot_intervention_results(performance, metric='accuracy', title_font=title_font, label_font=label_font, tick_font=tick_font, legend_font=legend_font)