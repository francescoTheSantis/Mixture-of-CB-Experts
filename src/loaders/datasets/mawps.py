import os
import re
import torch
import random
import pickle
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from env import DATA_PATH


MAWPS_DIR = f'{DATA_PATH}mawps'
# make the directory if it does not exist
if not os.path.exists(MAWPS_DIR):
    os.makedirs(MAWPS_DIR)
INPUT_COLUMN = 'Question'
TASK_NAMES = ['Answer']
CONCEPT_NAMES = ['N_00', 'N_01', 'N_02']



def is_linear_formula(formula: str) -> bool:
    try:
        expr = sp.sympify(formula)
        vars = expr.free_symbols
        poly = expr.as_poly(*vars)
        if poly is None:
            return False
        return poly.total_degree() == 1

    except Exception as e:
        print(f"Errore nell'analisi della formula: {e}")
        return False

def is_constant_formula(formula: str) -> bool:
    try:
        expr = sp.sympify(formula)
        vars = expr.free_symbols
        poly = expr.as_poly(*vars)
        if poly is None:
            return False 
        
        return poly.total_degree() == 0
    
    except Exception as e:
        print(f"Errore nell'analisi della formula: {e}")
        return False
    
def symbols_in_order(formula):
    pattern = r'N_\d+'
    matches = re.findall(pattern, formula)
    seen = set()
    ordered = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            ordered.append(sp.Symbol(m))
    return ordered
    
def standardize_formula(formula: str) -> str:
    '''
    Standardize the formula by replacing variables with x, y, z, w in order of appearance.
    E.g., if the formula is "a + b * c", it will be converted to "x + y * z".
    '''
    possible_vars = ['x', 'y', 'z', 'w', 'u']
    try:
        expr = sp.sympify(formula)
        symbols = symbols_in_order(formula)
        # maps the variables to x,y,z,w
        subs_dict = {str(symbol): possible_vars[i] for i, symbol in enumerate(symbols)}
        new_formula = formula
        for old_var, new_var in subs_dict.items():
            new_formula = re.sub(r'\b' + re.escape(old_var) + r'\b', new_var, new_formula)
        return new_formula
    except Exception as e:
        print(f"Error standardizing formula '{formula}': {e}")
        return formula
    
def histogram_of_formulas(formulas, name = 'hist'):
    formulas_hist = dict()
    for eq in formulas:
        if eq in formulas_hist:
            formulas_hist[eq] += 1
        else:
            formulas_hist[eq] = 1
    # plot the histogram
    plt.figure(figsize=(12, 6)) 
    plt.bar(formulas_hist.keys(), formulas_hist.values())
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.savefig(f'{MAWPS_DIR}/formula_{name}.png')
    plt.clf()
    return formulas_hist

def replace_values(values, answers, formulas, cap=5):
    ''' 
    Replace the values in the list with new random values between 0.1 and cap.
    values: list of list of floats
    answers: list of floats
    formulas: list of strings
    cap: float, greater than 50
    return: new_values, answers
    '''
    new_values = dict()
    new_answers = []
    for i in range(len(values)):
        v = values[i]
        formula = formulas[i]
        # check if any value is greater than cap

        new_values[i] = [round(random.uniform(0.1, 5), 2)
                            for n in v
                        ] # 0.1 to avoid zero division
        # create a dictionary to map N_0i to the values
        symbols = [sp.Symbol(f'N_0{j}') for j in range(len(new_values[i]))]
        # replace in the formula
        subs_dict = {s: val for s, val in zip(symbols, new_values[i])}

        try:
            # convertiamo la formula in oggetto sympy
            expr = sp.sympify(formula)
            # valutiamo la formula con i valori sostituiti
            result = float(expr.evalf(subs=subs_dict))
            new_answers.append(result)
            
        except Exception as e:
            print(f"Error evaluating formula '{formula}': {e}")
            return None

    return new_values, new_answers

def augment_data(df, augment_factor = 0.3, seed=42):
    # take some random rows
    n_rows_total = int(len(df)*augment_factor)
    new_df = pd.DataFrame()
    if augment_factor < 1:
        rows = df.sample(n=n_rows_total, random_state=seed)
        temp_rows = rows.copy().reset_index(drop=True)
        new_values, new_answers = replace_values(temp_rows['Numbers'].tolist(), 
                                                temp_rows['Answer'].tolist(), 
                                                temp_rows['Equation'].tolist())
        temp_rows['Numbers'] = [new_values[i] for i in range(len(new_values))]
        temp_rows['Answer'] = new_answers
        new_df = pd.concat([new_df, temp_rows], ignore_index=True)
    else:
        batch_size = min(32, n_rows_total-len(df))
        while len(new_df) < n_rows_total:
            rows = df.sample(n=batch_size, random_state=seed)
            temp_rows = rows.copy().reset_index(drop=True)
            new_values, new_answers = replace_values(temp_rows['Numbers'].tolist(), 
                                                    temp_rows['Answer'].tolist(), 
                                                    temp_rows['Equation'].tolist())
            temp_rows['Numbers'] = [new_values[i] for i in range(len(new_values))]
            temp_rows['Answer'] = new_answers
            new_df = pd.concat([new_df, temp_rows], ignore_index=True)

        if len(new_df) > n_rows_total:
            new_df = new_df.sample(n=n_rows_total, random_state=seed).reset_index(drop=True)

    # concatenate these new rows to the original dataframe
    df = pd.concat([df, new_df], ignore_index=True)
    # shuffle the dataframe
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df
    

def replace_N_with_values(question: str, values: list) -> str:
    '''
    Replace N_0i in the question with the corresponding value from values.
    '''
    new_question = question
    for i, val in enumerate(values):
        new_question = re.sub(r'\bN_0' + str(i) + r'\b', str(val), new_question)
    return new_question

def count_vars(expr):
    # Prendi solo le lettere x, y, z
    return len({v for v in expr if v in {'x', 'y', 'z'}})




class MAWPSDataset:
    def __init__(self,
                    already_created: bool = False,
                    batch_size: int = 128,
                    shuffle_seed = 42,
                    pre_trained_transformer="invokerliang/MWP-BERT-en"
                 ):

        self.name = "mawps"
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.concept_names = CONCEPT_NAMES
        self.pre_trained_transformer = pre_trained_transformer

        # Check if dataset files exist
        train_file = os.path.join(MAWPS_DIR, 'mawps_train.pkl')
        val_file = os.path.join(MAWPS_DIR, 'mawps_val.pkl')
        test_file = os.path.join(MAWPS_DIR, 'mawps_test.pkl')
        
        files_exist = os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file)
        
        # Create dataset if files don't exist or if explicitly requested
        if not files_exist or not already_created:

            # load the dataset
            ds = load_dataset("mwpt5/MAWPS")
            ds = ds['train']

            # identify constant and linear formulas
            ds = ds.add_column('Isconstant', [is_constant_formula(eq) for eq in ds['Equation']])
            ds = ds.add_column('Islinear', [is_linear_formula(eq) for eq in ds['Equation']])

            # eliminate constant functions
            ds = ds.filter(lambda example: not example['Isconstant'])

            # standardize formulas
            ds = ds.add_column('Standardized_Equation', [standardize_formula(eq) for eq in ds['Equation']])
            formulas_hist = histogram_of_formulas(ds['Standardized_Equation'])

            # keep only formulas that appear at least 60 times
            frequent_formulas = {eq for eq, count in formulas_hist.items() if count >= 60}

            # Remove formulas with 2 variables
            formulas_2_vars = [expr for expr in frequent_formulas if count_vars(expr) == 2]
            if len(formulas_2_vars) >0:
                formulas_2_vars = set(formulas_2_vars)
                frequent_formulas = frequent_formulas - formulas_2_vars

            # eliminate from frequent formulas all the linears except from one
            #linears = {eq for eq in frequent_formulas if is_linear_formula(eq)}
            #if len(linears) > 0:
            #    linears = list(linears)
            #    # order linears in base of their count in formulas_hist
            #    linears.sort(key=lambda x: formulas_hist[x], reverse=True)
            #    frequent_formulas = frequent_formulas - set(linears[:-1])
            ds = ds.filter(lambda example: example['Standardized_Equation'] in frequent_formulas)

            # check composition
            formulas_hist = histogram_of_formulas(ds['Standardized_Equation'])

            # cap numbers between 0.1 and 5 and update answers accordingly
            numbers = [list(map(float, num.split())) for num in ds['Numbers']]
            new_numbers, new_answers = replace_values(numbers, ds['Answer'], ds['Equation'])
            
            ds = ds.remove_columns('Numbers')
            ds = ds.add_column('Numbers', [new_numbers[i] for i in range(len(ds))])
            ds = ds.remove_columns('Answer')
            ds = ds.add_column('Answer', new_answers)
                   
            # divide in train, val, test stratifying by Standardized_Equation
            unique_formulas = list(formulas_hist.keys())
            for unique_form in unique_formulas:
                sub_ds = ds.filter(lambda example: example['Standardized_Equation'] == unique_form)
                sub_ds = sub_ds.shuffle(seed=self.shuffle_seed)
                n = len(sub_ds)
                n_train = int(0.5 * n)
                n_val = int(0.25 * n)
                train_ds_temp = sub_ds.select(range(n_train))
                val_ds_temp = sub_ds.select(range(n_train, n_train + n_val))
                test_ds_temp = sub_ds.select(range(n_train + n_val, n))

                # transform in pandas dataframe
                train_ds_temp = train_ds_temp.to_pandas()
                val_ds_temp = val_ds_temp.to_pandas()
                test_ds_temp = test_ds_temp.to_pandas()

                # concatenate the datasets
                if 'train_ds' in locals():
                    train_ds = pd.concat([train_ds, train_ds_temp], ignore_index=True)
                    val_ds = pd.concat([val_ds, val_ds_temp], ignore_index=True)
                    test_ds = pd.concat([test_ds, test_ds_temp], ignore_index=True)
                else:
                    train_ds = train_ds_temp
                    val_ds = val_ds_temp
                    test_ds = test_ds_temp

            # shuffle the datasets
            train_ds = train_ds.sample(frac=1, random_state=self.shuffle_seed).reset_index(drop=True)
            val_ds = val_ds.sample(frac=1, random_state=self.shuffle_seed).reset_index(drop=True)
            test_ds = test_ds.sample(frac=1, random_state=self.shuffle_seed).reset_index(drop=True)

            # check composition
            formulas_hist_train = histogram_of_formulas(train_ds['Standardized_Equation'], name = 'hist_train')
            formulas_hist_val = histogram_of_formulas(val_ds['Standardized_Equation'], name = 'hist_val')
            formulas_hist_test = histogram_of_formulas(test_ds['Standardized_Equation'], name = 'hist_test')

            # training data augmentation
            train_ds = augment_data(train_ds, augment_factor=150, seed=self.shuffle_seed)
            val_ds = augment_data(val_ds, augment_factor=100, seed=self.shuffle_seed)

            # replace N_0i with numbers in the Question column
            train_questions = [replace_N_with_values(q, nums) for q, nums in zip(train_ds['Question'], train_ds['Numbers'])]
            val_questions = [replace_N_with_values(q, nums) for q, nums in zip(val_ds['Question'], val_ds['Numbers'])]
            test_questions = [replace_N_with_values(q, nums) for q, nums in zip(test_ds['Question'], test_ds['Numbers'])]
            train_ds = train_ds.drop(columns=['Question'])
            val_ds = val_ds.drop(columns=['Question'])
            test_ds = test_ds.drop(columns=['Question'])
            train_ds = train_ds.assign(Question=train_questions)
            val_ds = val_ds.assign(Question=val_questions)
            test_ds = test_ds.assign(Question=test_questions)

            # create concepts columns with values from Numbers
            for i in range(3):
                train_ds[CONCEPT_NAMES[i]] = train_ds['Numbers'].apply(lambda x: x[i])
                val_ds[CONCEPT_NAMES[i]] = val_ds['Numbers'].apply(lambda x: x[i])
                test_ds[CONCEPT_NAMES[i]] = test_ds['Numbers'].apply(lambda x: x[i])

            # eliminate unnecessary columns
            train_ds = train_ds.drop(columns=['Isconstant', 'Islinear', 'Numbers'])
            val_ds = val_ds.drop(columns=['Isconstant', 'Islinear', 'Numbers'])
            test_ds = test_ds.drop(columns=['Isconstant', 'Islinear', 'Numbers'])

            formulas_hist_train = histogram_of_formulas(train_ds['Standardized_Equation'], name = 'hist_train')
            formulas_hist_val = histogram_of_formulas(val_ds['Standardized_Equation'], name = 'hist_val')
            formulas_hist_test = histogram_of_formulas(test_ds['Standardized_Equation'], name = 'hist_test')

            # save the datasets and equations in pickle format
            # write formulas in a text file
            with open(f'{MAWPS_DIR}/formulas.txt', 'w') as f:
                for formula in unique_formulas:
                    f.write(f"{formula}\n")

            train_ds.to_pickle(f'{MAWPS_DIR}/mawps_train.pkl')
            val_ds.to_pickle(f'{MAWPS_DIR}/mawps_val.pkl')
            test_ds.to_pickle(f'{MAWPS_DIR}/mawps_test.pkl')
            print(f"Datasets created and saved in {MAWPS_DIR}")

        else:
            print(f"Loading existing datasets from {MAWPS_DIR}")
        
        # load the datasets
        train_dataset = Dataset.from_pandas(pd.read_pickle(os.path.join(MAWPS_DIR, 'mawps_train.pkl')))
        val_dataset = Dataset.from_pandas(pd.read_pickle(os.path.join(MAWPS_DIR, 'mawps_val.pkl')))
        test_dataset = Dataset.from_pandas(pd.read_pickle(os.path.join(MAWPS_DIR, 'mawps_test.pkl')))

        pretrained_model_path = self.pre_trained_transformer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        
        tokenized_train = train_dataset.map(
            self.preprocess_function,
            batched=True
        )

        tokenized_val = val_dataset.map(
            self.preprocess_function,
            batched=True
        )

        tokenized_test = test_dataset.map(
            self.preprocess_function,
            batched=True
        )  
        self.train_dataset = tokenized_train
        self.val_dataset = tokenized_val
        self.test_dataset = tokenized_test

    def preprocess_function(self, examples):
        model_inputs = self.tokenizer(
            examples["Question"],
            truncation=True,
            padding = 'max_length',
            max_length=128
        )

        model_inputs["Answer"] = examples["Answer"]

        # now add the concepts
        for concept in self.concept_names:
            model_inputs[concept] = examples[concept]

        return model_inputs


    def collator(self):
        data_collator = CustomDataCollator()
        loaded_train = DataLoader(
            self.train_dataset, 
            collate_fn=data_collator, 
            batch_size=self.batch_size, 
            shuffle=True
            )

        loaded_val = DataLoader(
            self.val_dataset, 
            collate_fn=data_collator, 
            batch_size=self.batch_size, 
            shuffle=False
            )
        
        loaded_test = DataLoader(
            self.test_dataset, 
            collate_fn=data_collator, 
            batch_size=self.batch_size, 
            shuffle=False
            )
        
        return loaded_train, loaded_val, loaded_test


class CustomDataCollator:
    def __init__(self):
        self.concept_names = [concept for concept in CONCEPT_NAMES]
        self.task_names = TASK_NAMES
        self.input = [INPUT_COLUMN]

    def __call__(self, batch):

        # transform the batch into a tensor
        labels = torch.Tensor([[example[concept] for concept in self.task_names] for example in batch])
        if len(self.task_names) == 1:
            labels = labels.squeeze(1)

        concepts = torch.tensor(
            [[example[concept] for concept in self.concept_names] for example in batch], dtype=torch.float32
        )

        input_ids = torch.Tensor([example['input_ids'] for example in batch])
        #token_type_ids = torch.Tensor([example['token_type_ids'] for example in batch])
        attention_mask = torch.Tensor([example['attention_mask'] for example in batch])


        return {
            'x': {
                'input_ids': input_ids, 
                #'token_type_ids': token_type_ids, 
                'attention_mask': attention_mask
            },
            'c': concepts,
            'y': labels
        }
    




