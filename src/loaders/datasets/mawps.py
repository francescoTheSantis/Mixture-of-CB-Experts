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

from env import DATA_PATH, OPENAI_API_KEY

from tqdm import tqdm


MAWPS_DIR = os.path.join(DATA_PATH, 'mawps')
# make the directory if it does not exist
os.makedirs(MAWPS_DIR, exist_ok=True)
INPUT_COLUMN = 'Question'
TASK_NAMES = ['Answer']
CONCEPT_NAMES = ['N_00', 'N_01', 'N_02']

# Augmentation batch configuration
BALANCE_DATASET = False  # Whether to balance the dataset by generating new questions for underrepresented equations
QUESTIONS_PER_BATCH = 3  # Number of example questions to show to LLM per batch
NUM_BATCHES_PER_EQUATION = 2  # Number of batches to process for each equation
AUGMENTING_FACTOR = 0  # Number of new questions to generate per batch
NUMERICAL_AUGMENTING_FACTOR = 500  # Number of different numerical combinations per question (for training only)

# Training size after augmentation:
# if d is the size of the trianing set before augmentation,
# the size after augmentation will be:
#   [ d + ( AUGMENTING_FACTOR * NUM_BATCHES_PER_EQUATION ) ] * NUMERICAL_AUGMENTING_FACTOR

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

def augment_data(df, questions_per_batch=1, num_batches=1, seed=42):
    """
    Augment dataset using OpenAI GPT-4o to balance equation distribution.
    Counts frequency of each equation and generates additional questions for 
    underrepresented equations to match the most frequent one.
    LLM generates questions with N_0i placeholders.
    
    Args:
        df: DataFrame with columns ['Question', 'Equation']
        questions_per_batch: Number of example questions to show LLM per batch
        num_batches: Not used in balancing mode - kept for compatibility
        seed: Random seed for reproducibility
        balance: If False, returns original data without balancing.
    
    Returns:
        Augmented DataFrame with balanced equation distribution
    """

    columns_to_keep = ['Question', 'Equation']
    
    # Internal batch size for API calls
    GENERATION_BATCH_SIZE = 20

    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not set. Skipping augmentation.")
        return df
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Group questions by equation
    eq_dict = {}
    for idx, row in df.iterrows():
        equation = row['Equation']
        question = row['Question']
        if equation not in eq_dict:
            eq_dict[equation] = []
        eq_dict[equation].append(question)
    
    # Count frequency of each equation
    eq_frequencies = {eq: len(questions) for eq, questions in eq_dict.items()}
    max_frequency = max(eq_frequencies.values())
    
    print(f"Found {len(eq_dict)} unique equations")
    print(f"Equation frequencies: {eq_frequencies}")
    print(f"Maximum frequency: {max_frequency}")
    print(f"Balancing dataset to {max_frequency} samples per equation")
    
    new_rows = []
    total_to_generate = sum(max(0, max_frequency - freq) for freq in eq_frequencies.values())
    
    with tqdm(total=total_to_generate, desc="Generating questions for balancing") as pbar:
        for equation, questions in eq_dict.items():
            current_count = len(questions)
            needed = max_frequency - current_count
            
            if needed <= 0:
                continue  # This equation already has max frequency
            
            print(f"\nEquation '{equation}': has {current_count}, needs {needed} more")
            
            # Generate questions in batches
            remaining = needed
            random.seed(seed)
            
            while remaining > 0:
                # Determine batch size for this API call
                batch_size = min(remaining, GENERATION_BATCH_SIZE)
                
                # Sample questions for this batch
                if len(questions) <= questions_per_batch:
                    batch_questions = questions
                else:
                    batch_questions = random.sample(questions, questions_per_batch)
                
                # Format example questions
                examples_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(batch_questions)])
                
                # Create prompt for GPT-4o
                prompt = f"""You are a math problem generator. Given example math problems and their equation, generate {batch_size} DIFFERENT problems that require the SAME equation to solve.

Example questions:
{examples_str}

Equation used: {equation}

IMPORTANT RULES:
1. Generate {batch_size} completely NEW and DIFFERENT problems (different contexts, scenarios, objects)
2. Each problem MUST use exactly the same equation structure: {equation}
3. Use PLACEHOLDERS N_00, N_01, N_02 in your questions instead of actual numbers
4. The placeholders N_00, N_01, N_02 refer to the three numerical values in order of their appearance in the question
5. Make problems realistic and contextually diverse (different from the examples)
6. Do NOT include actual numerical values - only use the placeholders N_00, N_01, N_02

Provide your response as a JSON array with {batch_size} objects, each containing:
- "question": the new problem statement with N_00, N_01, N_02 placeholders

Example format:
[
  {{
    "question": "A baker made N_00 cookies on Monday and N_01 cookies on Tuesday. If he packages them in boxes of N_02 cookies each, how many boxes does he need?"
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
                        
                        # Create new row with only question and equation
                        new_row = {
                            'Question': new_question,
                            'Equation': equation
                        }
                        new_rows.append(new_row)
                        pbar.update(1)
                    
                    remaining -= len(generated_problems)
                    
                except Exception as e:
                    print(f"Error generating problems for equation '{equation}': {e}")
                    # Continue to next batch
                    remaining -= batch_size
    
    # Create DataFrame from new rows
    if new_rows:
        augmented_df = pd.DataFrame(new_rows)
        # Concatenate with original
        df = pd.concat([df[columns_to_keep], augmented_df[columns_to_keep]], ignore_index=True)
        print(f"\nSuccessfully generated {len(new_rows)} new samples")
        
        # Print final distribution
        final_eq_dict = {}
        for idx, row in df.iterrows():
            eq = row['Equation']
            if eq not in final_eq_dict:
                final_eq_dict[eq] = 0
            final_eq_dict[eq] += 1
        print(f"Final equation distribution: {final_eq_dict}")
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

        # load the dataset
        ds = load_dataset("mwpt5/MAWPS")
        ds = ds['train']

        # identify linear formulas
        ds = ds.add_column('Isconstant', [is_constant_formula(eq) for eq in ds['Equation']])
        ds = ds.add_column('Islinear', [is_linear_formula(eq) for eq in ds['Equation']])

        # eliminate constant functions
        ds = ds.filter(lambda example: not example['Isconstant'])

        ############## Automatic Equation filtering ##############
        # # Keep only formulas with 3 variables
        # formulas_3_vars = [expr for expr in formulas_hist.keys() if count_vars(expr) == 3]

        # # Eliminate formulas that do not have 3 variables
        # ds = ds.filter(lambda example: example['Equation'] in formulas_3_vars)

        # # Eliminate formulas that happear less than 30 times
        # frequent_formulas = {eq for eq, count in formulas_hist.items() if count >= 30}
        # ds = ds.filter(lambda example: example['Equation'] in frequent_formulas)
        ############## End of Automatic Equation filtering ##############

        ############## Manual Equation filtering ##############
        # We eliminated all the linear equations as we want to show how the symbolic version 
        # of our class of model is capable to obtain good results even on non-linear equations.
        equations_to_keep = [
            # "( N_00 + N_01 ) / N_02",
            # "( N_01 + N_02 ) / N_00",
            "N_02 * ( N_00 + N_01 )", # <-----
            "N_00 * ( N_01 - N_02 )", # <-----
            # "N_00 + N_02 - N_01",
            "N_02 * ( N_00 - N_01 )", # <-----
            # "N_00 + N_01 + N_02",
            # "N_00 + N_01 - N_02",
            # "( N_00 - N_01 ) / N_02", 
            "N_00 * ( N_01 + N_02 )", # <-----
        ]

        # Convert to pandas for processing
        ds = ds.to_pandas()
        ds = ds[ds['Equation'].isin(equations_to_keep)].reset_index(drop=True)
        ############## End of Manual Equation filtering ##############
        
        # compute dictionary of ocntaining key=Equation, value=count
        histogram_of_formulas(ds['Equation'], name='all', suffix='post_filtering')

        # Keep only Question and Equation columns
        ds = ds[['Question', 'Equation']]
        
        # data augmentation
        if BALANCE_DATASET:
            print("Augmenting dataset to balance equation distribution...")
            ds = augment_data(ds, 
                questions_per_batch=QUESTIONS_PER_BATCH,
                num_batches=NUM_BATCHES_PER_EQUATION,
                seed=self.shuffle_seed
            )
        
        ds = self._generate_numbers_and_answers(
            ds, 
            seed=self.shuffle_seed, 
            min_val=-4.0,
            max_val=4.0,
            numerical_augmentation_factor=NUMERICAL_AUGMENTING_FACTOR
        )

        # # Divide the dataset into train, val, test splits (80%, 10%, 10%)
        # # Stratify over the Equation
        train_ds, test_ds = train_test_split(ds, train_size=0.8, test_size=0.2, stratify=ds['Equation'], shuffle=True, random_state=self.shuffle_seed)
        val_ds, test_ds = train_test_split(test_ds, train_size=0.5, test_size=0.5, stratify=test_ds['Equation'], shuffle=True, random_state=self.shuffle_seed)

        # # shuffle the datasets
        # train_ds = train_ds.sample(frac=1, random_state=self.shuffle_seed).reset_index(drop=True)
        # val_ds = val_ds.sample(frac=1, random_state=self.shuffle_seed).reset_index(drop=True)
        # test_ds = test_ds.sample(frac=1, random_state=self.shuffle_seed).reset_index(drop=True)

        # Check if the set of equations are the same in all splits
        train_formulas = set(train_ds['Equation'].unique())
        val_formulas = set(val_ds['Equation'].unique())
        test_formulas = set(test_ds['Equation'].unique())
        unique_formulas = train_formulas.union(val_formulas).union(test_formulas)
        assert len(train_formulas & val_formulas & test_formulas) == len(unique_formulas), "Formulas differ across splits!"

        # save the datasets and equations in pickle format
        # write formulas in a text file (seed-specific)
        # with open(f'{MAWPS_DIR}/formulas_seed.txt', 'w') as f:
        #     for formula in unique_formulas:
        #         f.write(f"{formula}\n")

        # train_ds.to_pickle(f'{MAWPS_DIR}/mawps_train_seed{self.shuffle_seed}.pkl')
        # val_ds.to_pickle(f'{MAWPS_DIR}/mawps_val_seed{self.shuffle_seed}.pkl')
        # test_ds.to_pickle(f'{MAWPS_DIR}/mawps_test_seed{self.shuffle_seed}.pkl')
        
        # Save as CSV files as well
        # train_ds.to_csv(f'{MAWPS_DIR}/mawps_train_seed{self.shuffle_seed}.csv', index=False)
        # val_ds.to_csv(f'{MAWPS_DIR}/mawps_val_seed{self.shuffle_seed}.csv', index=False)
        # test_ds.to_csv(f'{MAWPS_DIR}/mawps_test_seed{self.shuffle_seed}.csv', index=False)

        # print(f"Final dataset sizes:")
        # print(f"  Train: {len(train_ds)} samples")
        # print(f"  Val: {len(val_ds)} samples")
        # print(f"  Test: {len(test_ds)} samples")

        # print(f"Datasets created and saved in {MAWPS_DIR} for seed {self.shuffle_seed}")

        # load the datasets (with seed-specific filenames)
        # Use preserve_index=False to ensure clean conversion without index issues
        # train_dataset = Dataset.from_pandas(pd.read_pickle(os.path.join(MAWPS_DIR, f'mawps_train_seed{self.shuffle_seed}.pkl')), preserve_index=False)
        # val_dataset = Dataset.from_pandas(pd.read_pickle(os.path.join(MAWPS_DIR, f'mawps_val_seed{self.shuffle_seed}.pkl')), preserve_index=False)
        # test_dataset = Dataset.from_pandas(pd.read_pickle(os.path.join(MAWPS_DIR, f'mawps_test_seed{self.shuffle_seed}.pkl')), preserve_index=False)

        # Use the configured pre-trained transformer as tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_transformer)

        # Check if the splits preserve the equation distribution
        histogram_of_formulas(train_ds['Equation'], name='train', suffix='post_augmentation')
        histogram_of_formulas(val_ds['Equation'], name='val', suffix='post_augmentation')
        histogram_of_formulas(test_ds['Equation'], name='test', suffix='post_augmentation')

        # Convert pandas DataFrames back to HuggingFace Datasets
        train_ds = Dataset.from_pandas(train_ds, preserve_index=False)
        val_ds = Dataset.from_pandas(val_ds, preserve_index=False)
        test_ds = Dataset.from_pandas(test_ds, preserve_index=False)

        # Shuffle the datasets
        train_ds = train_ds.shuffle(seed=self.shuffle_seed)
        val_ds = val_ds.shuffle(seed=self.shuffle_seed)
        test_ds = test_ds.shuffle(seed=self.shuffle_seed)

        # # Save the dataset as csv files
        # train_ds.to_csv(f'{MAWPS_DIR}/mawps_train_seed{self.shuffle_seed}.csv', index=False)
        # val_ds.to_csv(f'{MAWPS_DIR}/mawps_val_seed{self.shuffle_seed}.csv', index=False)
        # test_ds.to_csv(f'{MAWPS_DIR}/mawps_test_seed{self.shuffle_seed}.csv', index=False)

        # Use num_proc=1 to ensure deterministic processing order
        # Use load_from_cache_file=False to avoid stale cache issues
        tokenized_train = train_ds.map(
            self.preprocess_function,
            batched=True,
        )

        tokenized_val = val_ds.map(
            self.preprocess_function,
            batched=True,
        )

        tokenized_test = test_ds.map(
            self.preprocess_function,
            batched=True,
        )  

        self.train_dataset = tokenized_train
        self.val_dataset = tokenized_val
        self.test_dataset = tokenized_test

    def _generate_numbers_and_answers(self, df, seed=42, min_val=-1.0, max_val=1.0, numerical_augmentation_factor=1):
        """
        Generate random numbers for N_00, N_01, N_02 and compute Answer by evaluating the Equation.
        Also replaces placeholders in Question with the generated numbers.
        
        Args:
            df: DataFrame with columns ['Question', 'Equation']
            seed: Random seed for reproducibility
            min_val: Minimum value for generated numbers
            max_val: Maximum value for generated numbers
            numerical_augmentation_factor: Number of different numerical combinations to generate per question
        
        Returns:
            DataFrame with N_00, N_01, N_02, Answer columns and updated Question text
        """
        random.seed(seed)
        
        def get_denominator_variables(equation):
            """
            Identify which variables (N_00, N_01, N_02) appear in denominators.
            Returns a set of variable names that are used as denominators.
            """
            try:
                expr = sp.sympify(equation)
                denominators = set()
                
                # Walk through the expression tree to find divisions
                for arg in sp.preorder_traversal(expr):
                    if isinstance(arg, sp.Mul):
                        # Check for terms with negative powers (denominators)
                        for factor in arg.args:
                            if isinstance(factor, sp.Pow) and factor.exp.is_negative:
                                # Extract variables from this factor
                                for sym in factor.free_symbols:
                                    denominators.add(str(sym))
                    elif isinstance(arg, sp.Pow) and arg.exp.is_negative:
                        # Direct negative power
                        for sym in arg.free_symbols:
                            denominators.add(str(sym))
                
                return denominators
            except Exception as e:
                print(f"Error parsing equation '{equation}': {e}")
                return set()
        
        def generate_safe_number(min_val, max_val, is_denominator):
            """
            Generate a random number. If it's for a denominator, ensure it's not in [-0.5, 0.5].
            """
            value = round(random.uniform(min_val, max_val), 2)
            if is_denominator:
                # Regenerate until we get a value outside [-0.1, 0.1]
                while -0.1 <= value <= 0.1:
                    value = round(random.uniform(min_val, max_val), 2)
            return value
        
        n00_list, n01_list, n02_list = [], [], []
        answers = []
        questions_with_values = []
        equations_list = []
        
        for idx, row in df.iterrows():
            equation = row['Equation']
            question = row['Question']
            
            # Identify which variables are in denominators
            denominator_vars = get_denominator_variables(equation)
            
            # Generate multiple numerical combinations for this question
            for aug_idx in range(numerical_augmentation_factor):
                # Generate random numbers, avoiding small values for denominators
                numbers = [
                    generate_safe_number(min_val, max_val, 'N_00' in denominator_vars),
                    generate_safe_number(min_val, max_val, 'N_01' in denominator_vars),
                    generate_safe_number(min_val, max_val, 'N_02' in denominator_vars)
                ]
                n00_list.append(numbers[0])
                n01_list.append(numbers[1])
                n02_list.append(numbers[2])
                
                # Replace placeholders in the question
                question_with_values = replace_N_with_values(question, numbers)
                questions_with_values.append(question_with_values)
                equations_list.append(equation)
                
                # Calculate answer using the equation
                symbols = [sp.Symbol(f'N_0{j}') for j in range(3)]
                subs_dict = {s: val for s, val in zip(symbols, numbers)}
                
                try:
                    expr = sp.sympify(equation)
                    answer = float(expr.evalf(subs=subs_dict))
                    answers.append(answer)
                except Exception as e:
                    print(f"Error evaluating equation '{equation}' with values {numbers}: {e}")
                    answers.append(0.0)  # Default value in case of error
        
        # Create new dataframe with all augmented samples
        df_augmented = pd.DataFrame({
            'Question': questions_with_values,
            'Equation': equations_list,
            'N_00': n00_list,
            'N_01': n01_list,
            'N_02': n02_list,
            'Answer': answers
        })
        
        return df_augmented
    
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
        
        # Create a generator with a fixed seed for reproducible shuffling
        generator = torch.Generator()
        generator.manual_seed(self.shuffle_seed)
        
        loaded_train = DataLoader(
            self.train_dataset, 
            collate_fn=data_collator, 
            batch_size=self.batch_size, 
            shuffle=True,
            generator=generator
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
        attention_mask = torch.Tensor([example['attention_mask'] for example in batch])

        # Include the raw question text for BERT embedding extraction
        questions = [example.get('Question', '') for example in batch]

        # Build the result dict
        result = {
            'x': {
                'input_ids': input_ids, 
                'attention_mask': attention_mask
            },
            'c': concepts,
            'y': labels,
            'questions': questions  # Add raw text for BERT preprocessing
        }
        
        # Add token_type_ids only if present (T5 models don't use them)
        if 'token_type_ids' in batch[0]:
            token_type_ids = torch.Tensor([example['token_type_ids'] for example in batch])
            result['x']['token_type_ids'] = token_type_ids
        
        return result




