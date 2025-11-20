import os
import re
import torch
import random
import pickle
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from datasets import load_dataset, Dataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from openai import OpenAI
from sklearn.model_selection import train_test_split

try:
    from env import DATA_PATH, OPENAI_API_KEY
except:
    import sys
    from pathlib import Path
    # Add the project root to path (3 levels up from this file)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from env import DATA_PATH, OPENAI_API_KEY
from tqdm import tqdm


MAWPS_DIR = os.path.join(DATA_PATH, 'mawps')
# make the directory if it does not exist
os.makedirs(MAWPS_DIR, exist_ok=True)
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
    
def histogram_of_formulas(formulas, name = 'hist', suffix=''):
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
    plt.savefig(f'{MAWPS_DIR}/formula_{name}_{suffix}.pdf')
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

def augment_data(df, augmenting_factor=10, seed=42):
    """
    Augment dataset using OpenAI GPT-4o to generate similar math problems.
    For each sample, generates augmenting_factor new questions that use the same equation.
    
    Args:
        df: DataFrame with columns ['Question', 'Equation', 'Standardized_Equation', 'Answer']
        augmenting_factor: Number of new questions to generate per original sample
        seed: Random seed for reproducibility
    
    Returns:
        Augmented DataFrame with original + generated samples
    """

    columns_to_keep = ['Question', 'Equation', 'Answer', 'N_00', 'N_01', 'N_02']

    if augmenting_factor == 0:
        return df[columns_to_keep]

    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not set. Skipping augmentation.")
        return df
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    new_rows = []
    
    print(f"Augmenting dataset with {augmenting_factor} new questions per sample...")
    print(f"Total samples to process: {len(df)}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting samples"):
        
        original_question = row['Question']
        equation = row['Equation']
        
        # Calculate range hints for the LLM based on original numbers
        n0_range = f"{max(0.1, row['N_00'] * 0.5):.1f} to {row['N_00'] * 2:.1f}"
        n1_range = f"{max(0.1, row['N_01'] * 0.5):.1f} to {row['N_01'] * 2:.1f}"
        n2_range = f"{max(0.1, row['N_02'] * 0.5):.1f} to {row['N_02'] * 2:.1f}"
        
        # Create prompt for GPT-4o
        prompt = f"""You are a math problem generator. Given an original math problem and its equation, generate {augmenting_factor} DIFFERENT problems that require the SAME equation to solve.

Original question: "{original_question}"
Equation used: {equation}
Original numbers: N_00={row['N_00']:.2f}, N_01={row['N_01']:.2f}, N_02={row['N_02']:.2f}

IMPORTANT RULES:
1. Generate {augmenting_factor} completely NEW and DIFFERENT problems (different contexts, scenarios, objects)
2. Each problem MUST use exactly the same equation structure: {equation}
3. The numbers N_00, N_01, N_02 refer to the three numerical values where the index indicates their order of appearance in the question.
4. Provide 3 positive numbers in SIMILAR RANGES to the original sample:
   - N_00 should be in range: {n0_range}
   - N_01 should be in range: {n1_range}
   - N_02 should be in range: {n2_range}
5. Numbers MUST be real or integer values (NO NaN, NO infinity, NO null values)
6. Make problems realistic and contextually diverse (different from the original)

Provide your response as a JSON array with {augmenting_factor} objects, each containing:
- "question": the new problem statement
- "numbers": array of exactly 3 valid numeric values (real or integer, no NaN)

Example format:
[
  {{
    "question": "A baker made 24.5 cookies on Monday and 18.3 cookies on Tuesday. If he packages them in boxes of 6.0 cookies each, how many boxes does he need?",
    "numbers": [24.5, 18.3, 6.0]
  }}
]

Provide ONLY the JSON array, no additional text."""
        
        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates math problems in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,  # Higher temperature for more diversity
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            # Parse JSON response
            generated_problems = json.loads(response_text)
            
            # Process each generated problem
            for problem in generated_problems:
                new_question = problem['question']
                numbers = problem['numbers']
                
                if len(numbers) != 3:
                    print(f"Warning: Skipping problem with {len(numbers)} numbers (expected 3)")
                    continue
                
                # Calculate answer using the equation
                symbols = [sp.Symbol(f'N_0{j}') for j in range(3)]
                subs_dict = {s: val for s, val in zip(symbols, numbers)}
                
                try:
                    expr = sp.sympify(equation)
                    answer = float(expr.evalf(subs=subs_dict))
                    
                    # Create new row
                    new_row = {
                        'Question': new_question,
                        'Equation': equation,
                        'Answer': answer,
                        'N_00': numbers[0],
                        'N_01': numbers[1],
                        'N_02': numbers[2]
                    }
                    new_rows.append(new_row)
                    
                except Exception as e:
                    print(f"Error calculating answer for generated problem: {e}")
                    continue
            
            # # Rate limiting: sleep to avoid hitting API limits
            # time.sleep(0.5)
            
        except Exception as e:
            print(f"Error generating problems for sample {idx}: {e}")
            continue
    
    # Create DataFrame from new rows
    if new_rows:
        augmented_df = pd.DataFrame(new_rows)
        # Concatenate with original
        df = pd.concat([df[columns_to_keep], augmented_df[columns_to_keep]], ignore_index=True)
        print(f"Successfully generated {len(new_rows)} new samples")
    else:
        print("No new samples were generated")
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
    # check how many different variables are in the expression
    symbols = symbols_in_order(expr)
    return len(symbols)




class MAWPSDataset:
    def __init__(self,
                    already_created: bool = False,
                    batch_size: int = 128,
                    shuffle_seed = 42,
                    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                    pre_trained_transformer: str = 'bert-base-uncased'
                 ):

        self.name = "mawps"
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.concept_names = CONCEPT_NAMES
        self.device = device

        # Check if dataset files exist for this specific seed
        train_file = os.path.join(MAWPS_DIR, f'mawps_train_seed{shuffle_seed}.pkl')
        val_file = os.path.join(MAWPS_DIR, f'mawps_val_seed{shuffle_seed}.pkl')
        test_file = os.path.join(MAWPS_DIR, f'mawps_test_seed{shuffle_seed}.pkl')
        
        files_exist = os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file)
        
        # Create dataset if files don't exist or if explicitly requested
        if not files_exist or not already_created:

            # load the dataset
            ds = load_dataset("mwpt5/MAWPS")
            ds = ds['train']

            # identify linear formulas
            ds = ds.add_column('Isconstant', [is_constant_formula(eq) for eq in ds['Equation']])
            ds = ds.add_column('Islinear', [is_linear_formula(eq) for eq in ds['Equation']])

            # eliminate constant functions
            ds = ds.filter(lambda example: not example['Isconstant'])

            # compute dictionary of ocntaining key=Equation, value=count
            formulas_hist = histogram_of_formulas(ds['Equation'], name='all', suffix='pre_filtering')

            # Keep only formulas with 3 variables
            formulas_3_vars = [expr for expr in formulas_hist.keys() if count_vars(expr) == 3]

            # Eliminate formulas that do not have 3 variables
            ds = ds.filter(lambda example: example['Equation'] in formulas_3_vars)

            # Eliminate formulas that happear less than 30 times
            frequent_formulas = {eq for eq, count in formulas_hist.items() if count >= 30}
            ds = ds.filter(lambda example: example['Equation'] in frequent_formulas)

            # Divide the dataset into train, val, test splits (70%, 10%, 20%)
            # Stratify over the Equation
            ds = ds.to_pandas() 
            train_ds, test_ds = train_test_split(ds, train_size=0.7, test_size=0.3, stratify=ds['Equation'], shuffle=True)
            val_ds, test_ds = train_test_split(test_ds, train_size=(1/3), test_size=(2/3), stratify=test_ds['Equation'], shuffle=True)
            
            # Check if the splits preserve the equation distribution
            histogram_of_formulas(train_ds['Equation'], name='train', suffix='pre_augmentation')
            histogram_of_formulas(val_ds['Equation'], name='val', suffix='pre_augmentation')
            histogram_of_formulas(test_ds['Equation'], name='test', suffix='pre_augmentation')

            # Create concepts columns with values from Numbers (do this before replacing in questions)
            # Check if N_00, N_01, N_02 already exist (from augmentation) or need to be created from Numbers
            if 'N_00' not in train_ds.columns:
                for i in range(3):
                    train_ds[CONCEPT_NAMES[i]] = train_ds['Numbers'].apply(lambda x: float(x.split()[i]))
            if 'N_00' not in val_ds.columns:
                for i in range(3):
                    val_ds[CONCEPT_NAMES[i]] = val_ds['Numbers'].apply(lambda x: float(x.split()[i]))
            if 'N_00' not in test_ds.columns:
                for i in range(3):
                    test_ds[CONCEPT_NAMES[i]] = test_ds['Numbers'].apply(lambda x: float(x.split()[i]))

            # Replace N_0i with numbers in the Question column
            train_questions = [replace_N_with_values(q, [row['N_00'], row['N_01'], row['N_02']]) 
                             for idx, row in train_ds.iterrows() for q in [row['Question']]]
            val_questions = [replace_N_with_values(q, [row['N_00'], row['N_01'], row['N_02']]) 
                           for idx, row in val_ds.iterrows() for q in [row['Question']]]
            test_questions = [replace_N_with_values(q, [row['N_00'], row['N_01'], row['N_02']]) 
                            for idx, row in test_ds.iterrows() for q in [row['Question']]]
            
            train_ds = train_ds.drop(columns=['Question'])
            val_ds = val_ds.drop(columns=['Question'])
            test_ds = test_ds.drop(columns=['Question'])
            train_ds = train_ds.assign(Question=train_questions)
            val_ds = val_ds.assign(Question=val_questions)
            test_ds = test_ds.assign(Question=test_questions)
            
            # training data augmentation
            train_ds = augment_data(train_ds, augmenting_factor=10, seed=self.shuffle_seed)
            histogram_of_formulas(train_ds['Equation'], name='train', suffix='post_augmentation')

            # shuffle the datasets
            train_ds = train_ds.sample(frac=1, random_state=self.shuffle_seed).reset_index(drop=True)
            val_ds = val_ds.sample(frac=1, random_state=self.shuffle_seed).reset_index(drop=True)
            test_ds = test_ds.sample(frac=1, random_state=self.shuffle_seed).reset_index(drop=True)

            # Check if the set of equations are the same in all splits
            train_formulas = set(train_ds['Equation'].unique())
            val_formulas = set(val_ds['Equation'].unique())
            test_formulas = set(test_ds['Equation'].unique())
            unique_formulas = train_formulas.union(val_formulas).union(test_formulas)
            assert len(train_formulas & val_formulas & test_formulas) == len(unique_formulas), "Formulas differ across splits!"

            # save the datasets and equations in pickle format
            # write formulas in a text file (seed-specific)
            with open(f'{MAWPS_DIR}/formulas_seed{self.shuffle_seed}.txt', 'w') as f:
                for formula in unique_formulas:
                    f.write(f"{formula}\n")

            train_ds.to_pickle(f'{MAWPS_DIR}/mawps_train_seed{self.shuffle_seed}.pkl')
            val_ds.to_pickle(f'{MAWPS_DIR}/mawps_val_seed{self.shuffle_seed}.pkl')
            test_ds.to_pickle(f'{MAWPS_DIR}/mawps_test_seed{self.shuffle_seed}.pkl')
            
            # Save as CSV files as well
            train_ds.to_csv(f'{MAWPS_DIR}/mawps_train_seed{self.shuffle_seed}.csv', index=False)
            val_ds.to_csv(f'{MAWPS_DIR}/mawps_val_seed{self.shuffle_seed}.csv', index=False)
            test_ds.to_csv(f'{MAWPS_DIR}/mawps_test_seed{self.shuffle_seed}.csv', index=False)
            
            print(f"Datasets created and saved in {MAWPS_DIR} for seed {self.shuffle_seed}")

        else:
            print(f"Loading existing datasets from {MAWPS_DIR} for seed {self.shuffle_seed}")
        
        # load the datasets (with seed-specific filenames)
        train_dataset = Dataset.from_pandas(pd.read_pickle(os.path.join(MAWPS_DIR, f'mawps_train_seed{self.shuffle_seed}.pkl')))
        val_dataset = Dataset.from_pandas(pd.read_pickle(os.path.join(MAWPS_DIR, f'mawps_val_seed{self.shuffle_seed}.pkl')))
        test_dataset = Dataset.from_pandas(pd.read_pickle(os.path.join(MAWPS_DIR, f'mawps_test_seed{self.shuffle_seed}.pkl')))

        # Store BERT configuration for later use (will be used in preprocessing)
        # The actual BERT pretraining and embedding extraction happens in preprocessing.py
        
        # Use the configured pre-trained transformer as tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_transformer)
        
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

    def _save_samples_to_txt(self, df, split_name):
        """
        Save all samples from a split to a text file.
        Each sample includes: question, variable values (N_00, N_01, N_02), and answer.
        """
        output_file = os.path.join(MAWPS_DIR, f'mawps_{split_name}_samples.txt')
        
        with open(output_file, 'w') as f:
            f.write(f"MAWPS Dataset - {split_name.upper()} Split\n")
            f.write("=" * 80 + "\n\n")
            
            for idx, row in df.iterrows():
                f.write(f"Sample {idx + 1}:\n")
                f.write(f"Question: {row['Question']}\n")
                f.write(f"Variables:\n")
                f.write(f"  N_00 = {row['N_00']:.2f}\n")
                f.write(f"  N_01 = {row['N_01']:.2f}\n")
                f.write(f"  N_02 = {row['N_02']:.2f}\n")
                f.write(f"Answer: {row['Answer']:.2f}\n")
                f.write(f"Equation: {row['Equation']}\n")
                f.write(f"Standardized Equation: {row['Standardized_Equation']}\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"Saved {len(df)} samples to {output_file}")

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
        token_type_ids = torch.Tensor([example['token_type_ids'] for example in batch])
        attention_mask = torch.Tensor([example['attention_mask'] for example in batch])

        # Include the raw question text for BERT embedding extraction
        questions = [example.get('Question', '') for example in batch]

        return {
            'x': {
                'input_ids': input_ids, 
                'token_type_ids': token_type_ids, 
                'attention_mask': attention_mask
            },
            'c': concepts,
            'y': labels,
            'questions': questions  # Add raw text for BERT preprocessing
        }


if __name__ == "__main__":
    # Test the augment_data function
    print("Testing augment_data function...")
    
    # Create a small test dataframe with a few samples
    test_data = {
        'Question': [
            'At the town carnival Oliver rode the ferris wheel N_00 times and the bumper cars N_01 times . If each ride cost N_02 tickets , how many tickets did he use ?',
            'In one week , an airplane pilot flew N_00 miles on Tuesday and N_01 miles on Thursday . If the pilot flies the same number of miles N_02 weeks in a row , how many miles does the pilot fly in all ?'
        ],
        'Equation': [
            'N_02 * ( N_00 + N_01 )',
            'N_02 * ( N_00 + N_01 )'
        ],
        'Standardized_Equation': [
            'z * ( x + y )',
            'z * ( x + y )'
        ],
        'Answer': [45.0, 210.0],
        'N_00': [3.0, 50.0],
        'N_01': [6.0, 20.0],
        'N_02': [5.0, 3.0]
    }
    
    test_df = pd.DataFrame(test_data)
    
    print("\nOriginal DataFrame:")
    print(test_df)
    print(f"\nOriginal size: {len(test_df)} samples")
    
    augmenting_factor = 10
    print(f"\nRunning augmentation with augmenting_factor={augmenting_factor}...")
    augmented_df = augment_data(test_df, augmenting_factor=augmenting_factor, seed=42)
    
    print(f"\nAugmented size: {len(augmented_df)} samples")
    print(f"New samples generated: {len(augmented_df) - len(test_df)}")
    
    print("\nAugmented DataFrame:")
    print(augmented_df)
    
    # Save results to CSV for inspection
    output_file = f'{MAWPS_DIR}/test_augmentation.csv'
    augmented_df.to_csv(output_file, index=False)
    print(f"\nTest results saved to: {output_file}")
    
    # Display a few generated samples
    print("\n" + "="*80)
    print("Sample Generated Questions:")
    print("="*80)
    new_samples = augmented_df.tail(min(4, len(augmented_df) - len(test_df)))
    for idx, row in new_samples.iterrows():
        print(f"\nSample {idx + 1}:")
        print(f"Question: {row['Question']}")
        print(f"Numbers: N_00={row['N_00']:.2f}, N_01={row['N_01']:.2f}, N_02={row['N_02']:.2f}")
        print(f"Equation: {row['Equation']}")
        print(f"Answer: {row['Answer']:.2f}")
        print("-" * 80)




