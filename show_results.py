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
custom_order = ['xor', \
                #'dot', \
                #'checkmark', \
                #'trigonometry', \
                'mnist_addition', \
                'cub', \
                'awa2',
                'awa2_incomplete',
                'cub_incomplete',
                'cebab',
                #'celeba'
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
    'm_licem': {'marker': 's', 'name': 'M-LICEM', 'color': 'tab:green', 'size': marker_size},
    'pred_cbm': {'marker': 'P', 'name': 'Pred-CBM', 'color': 'tab:red', 'size': marker_size},
    'pred_cbm_local': {'marker': 'P', 'name': 'Pred-CBM Local', 'color': 'tab:blue', 'size': marker_size}    
}

# Call the function with the desired metric and font properties
legend_font = {'size': 44}
title_font = {'size': 36, 'weight': 'bold'}
label_font = {'size': 44}
tick_font = {'size': 28}

def main():

    try:
        # List the paths containing the results
        paths = [
            "/home/fdesantis/projects/Linear-Memory-Reasoner/output/blackbox/2025-08-25_18-52-25",
            "/home/fdesantis/projects/Linear-Memory-Reasoner/output/cbm_linear/2025-08-25_18-53-03",
            "/home/fdesantis/projects/Linear-Memory-Reasoner/output/cem/2025-08-25_18-54-03",
            "/home/fdesantis/projects/Linear-Memory-Reasoner/output/licem/2025-08-25_22-13-17",
            "/home/fdesantis/projects/Linear-Memory-Reasoner/output/m_licem/2025-08-25_22-13-17",
            "/home/fdesantis/projects/Linear-Memory-Reasoner/output/pred_cbm/2025-08-25_22-13-17",
            "/home/fdesantis/projects/Linear-Memory-Reasoner/output/cmr/2025-08-26_19-02-12",
            "/home/fdesantis/projects/Linear-Memory-Reasoner/output/dcr/2025-08-26_19-02-21",
            "/home/fdesantis/projects/Linear-Memory-Reasoner/output/pred_cbm_local/2025-08-27_18-46-37"
        ]

        result_figs = "figs"
        os.makedirs(result_figs, exist_ok=True)

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
                    # if 'test_task_acc' and 'test_concept_acc' are not in the dataframe, skip the experiment
                    if 'test_task_acc' not in result.columns:
                        d['task'] = result['test_task_mse'].iloc[-1]
                    else:
                        d['task'] = result['test_task_acc'].iloc[-1]

                    if conf['model']['metadata']['name']=='blackbox':
                        d['concept'] = 0
                    else:
                        if 'test_concept_acc' not in result.columns:
                            d['concept'] = result['test_concept_mse'].iloc[-1]
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

        # Count number of seeds
        num_seeds = performance['seed'].unique().max()
        print(f"Number of unique seeds: {num_seeds}")

        # Filter the performance dataframe to keep only the models in model_styles 
        # and datasets in custom_order.
        performance = performance[performance['model'].isin(model_styles.keys()) & \
                                performance['dataset'].isin(custom_order)]

        ######### Only for the LinearMemoryReasoner with seed=1, plot explanations #########
        # plot_explanations(lmr_paths)

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

        # If the dataset = 'cebab', then divide by 100
        task_stats.loc[task_stats['dataset'] == 'cebab', 'avg_accuracy_task'] /= 100
        task_stats.loc[task_stats['dataset'] == 'cebab', 'std_accuracy_task'] /= 100
        concept_stats.loc[concept_stats['dataset'] == 'cebab', 'avg_accuracy_concept'] /= 100
        concept_stats.loc[concept_stats['dataset'] == 'cebab', 'std_accuracy_concept'] /= 100

        # Merge the two DataFrames on 'model' and 'dataset'
        merged_stats = pd.merge(task_stats, concept_stats, on=['model', 'dataset'])

        merged_stats = merged_stats.sort_values('dataset')
        merged_stats['dataset'] = pd.Categorical(merged_stats['dataset'], categories=custom_order, ordered=True)

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
                # From std to SE
                std = 1.96 * (pivot_table_std.loc[i, j]*100) / num_seeds
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
                # From std to SE
                std = 1.96 * (pivot_table_std.loc[i, j]*100) / num_seeds
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
                    d = pd.read_csv(result_file)[['noise','p_int','f1','accuracy','mse']]
                
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
        print(f"Error occurred while showing results: {e}")


    ##### Visualize ablation over the memory. #######

    try:
        paths = [
            "/home/fdesantis/projects/Linear-Memory-Reasoner/output/memory_ablation/2025-08-28_11-09-32",
        ]

        result_figs = "figs"
        os.makedirs(result_figs, exist_ok=True)

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
                    d['memory_size'] = conf['memory_size']

                    with open(result_file, 'r') as file:
                        result = pd.read_csv(file, header=0)

                    # Select the last row of the dataframe where we test the model
                    # if 'test_task_acc' and 'test_concept_acc' are not in the dataframe, skip the experiment
                    if 'test_task_acc' not in result.columns:
                        d['task'] = result['test_task_mse'].iloc[-1]
                    else:
                        d['task'] = result['test_task_acc'].iloc[-1]

                    if conf['model']['metadata']['name']=='blackbox':
                        d['concept'] = 0
                    else:
                        if 'test_concept_acc' not in result.columns:
                            d['concept'] = result['test_concept_mse'].iloc[-1]
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

        # Show a subplot for each dataset. 
        # In each subplot there are bars as long as the accuracy reached by the models.
        # For each model there will be several lines, each one representing a different model's performance. 
        # The x-axis will represent the memory_size, while the y-axis will represent the task accuracy.
        plot_memory_ablation(performance, model_styles, title_font, label_font, tick_font)
    
    except Exception as e:
        print(f"Error occurred while plotting memory ablation results: {e}")


if __name__ == "__main__":
    main()