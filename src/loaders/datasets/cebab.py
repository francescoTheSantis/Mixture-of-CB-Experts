from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
import numpy as np
import ast 

class CEBaBDataset:
    def __init__(self, 
                 pre_trained_transformer='bert-base-uncased',
                 batch_size=32,
                 ):
        
        ds = load_dataset("CEBaB/CEBaB")
        self.batch_size = batch_size

        # get the relevant splits from the dataset (train observational contains only the original reviews which have not been modified by the annotators)
        ds_train = ds['train_observational']
        ds_val = ds['validation']
        ds_test = ds['test']

        # define concept names
        self.concept_names = ['food_aspect_label_distribution', 
                              'ambiance_aspect_label_distribution', 
                              'service_aspect_label_distribution', 
                              'noise_aspect_label_distribution']

        # convert to pandas
        ds_train = ds_train.to_pandas()
        ds_val = ds_val.to_pandas()
        ds_test = ds_test.to_pandas()

        selected_columns = ['description', 'review_label_distribution'] + self.concept_names

        # select only the relevant columns: review_label_distribution, food_aspect_label_distribution, 
        # ambiance_aspect_label_distribution, service_aspect_label_distribution, noise_aspect_label_distribution.
        ds_train = ds_train[selected_columns]
        ds_val = ds_val[selected_columns]
        ds_test = ds_test[selected_columns]

        # eliminate rows with missing values
        ds_train = ds_train.dropna()
        ds_val = ds_val.dropna()
        ds_test = ds_test.dropna()

        dss = [ds_train, ds_val, ds_test]

        for ds in dss:
            # the review and concepts have to be processed such that we compute, and then normalize, 
            # the mean of the values in the _label_distribution.
            # The NaN values can be produced by the get_normalized_mean function if the dictionary 
            # is empty or if there is an error in the conversion.
            ds['review'] = ds.apply(lambda row: self.get_normalized_mean(row['review_label_distribution'], 'review', True), axis=1)
            ds['food'] = ds.apply(lambda row: self.get_normalized_mean(row['food_aspect_label_distribution'], 'concept', True), axis=1)
            ds['ambiance'] = ds.apply(lambda row: self.get_normalized_mean(row['ambiance_aspect_label_distribution'], 'concept', True), axis=1)
            ds['service'] = ds.apply(lambda row: self.get_normalized_mean(row['service_aspect_label_distribution'], 'concept', True), axis=1)
            ds['noise'] = ds.apply(lambda row: self.get_normalized_mean(row['noise_aspect_label_distribution'], 'concept', True), axis=1)
            # drop the original label distributions
            ds = ds[['description', 'review', 'food', 'ambiance', 'service', 'noise']]

        # The NaN values can be produced by the get_normalized_mean function if the dictionary 
        # is empty or if there is an error in the conversion.
        ds_train = dss[0].dropna()
        ds_val = dss[1].dropna()
        ds_test = dss[2].dropna()

        # update the concept names to match the new columns
        self.concept_names = ['food', 'ambiance', 'service', 'noise']

        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_transformer)

        data_train = Dataset.from_pandas(ds_train)
        data_val = Dataset.from_pandas(ds_val)
        data_test = Dataset.from_pandas(ds_test)

        tokenized_train = data_train.map(
            self.preprocess_function,
            batched=True,
            remove_columns=data_train.column_names,
        )

        tokenized_val = data_val.map(
            self.preprocess_function,
            batched=True,
            remove_columns=data_val.column_names,
        )

        tokenized_test = data_test.map(
            self.preprocess_function,
            batched=True,
            remove_columns=data_test.column_names,
        )       

        self.tokenized_train = tokenized_train
        self.tokenized_val = tokenized_val
        self.tokenized_test = tokenized_test

    def mapping(self, key):
        """
        Map the key to an integer value.
        """
        if key=='Negative':
            return 1
        elif key=='unknown':
            return 2
        elif key=='Positive':
            return 3

    def get_normalized_mean(self, dictionary=None, type='review', normalize=False):
        """
        Given a dictionary of key-value pairs, where the keys are integers representing the value
        while the value is the amount of times that value appears, return the mean of the values.
        """
        # transform the string into a dictionary
        if dictionary is None or len(dictionary) == 0:
            return np.nan
        dictionary = ast.literal_eval(dictionary)
        if type == 'concept':
            # for concepts, we can just take the mean of the values
            dictionary = {self.mapping(key): value for key, value in dictionary.items()}
        try:
            total = sum(int(key) * int(value) for key, value in dictionary.items())
            count = sum([int(x) for x in dictionary.values()])
            mean = total / count if count > 0 else 0
            if type == 'review':
                if normalize:
                    # Knowing that the range is between 1 and 5, normalize the mean such that it sampled from a normal distribution with mean 0 and std 1.
                    normalized_mean = (mean - 3) / 2  # Normalize to mean 0, std 1
                else:
                    normalized_mean = mean
            else:
                if normalize:
                    # For concepts
                    normalized_mean = (mean - 1) / 2  # Normalize to mean 0, std 1
                else:
                    normalized_mean = mean
        except:
            normalized_mean = np.nan  
        return normalized_mean

    def preprocess_function(self, examples):
        model_inputs = self.tokenizer(
            examples["description"],
            truncation=True,
            padding = 'max_length'
        )
        model_inputs["review"] = examples["review"]

        # now add the concepts
        for concept in self.concept_names:
            model_inputs[concept] = examples[concept]

        return model_inputs

    def collator(self):
        data_collator = CustomDataCollator()
        loaded_train = DataLoader(
            self.tokenized_train, 
            collate_fn=data_collator, 
            batch_size=self.batch_size, 
            shuffle=True
            ) 
        
        loaded_val = DataLoader(
            self.tokenized_val, 
            collate_fn=data_collator, 
            batch_size=self.batch_size, 
            shuffle=False
            )
        
        loaded_test = DataLoader(
            self.tokenized_test, 
            collate_fn=data_collator, 
            batch_size=self.batch_size, 
            shuffle=False
            )
        
        return loaded_train, loaded_val, loaded_test

class CustomDataCollator:
    def __init__(self):
        pass
        
    def __call__(self, batch):

        # transform the batch into a tensor
        input_ids = torch.Tensor([example['input_ids'] for example in batch])
        token_type_ids = torch.Tensor([example['token_type_ids'] for example in batch])
        attention_mask = torch.Tensor([example['attention_mask'] for example in batch])
        labels = torch.Tensor([example['review'] for example in batch])
        food = torch.Tensor([example['food'] for example in batch])
        ambiance = torch.Tensor([example['ambiance'] for example in batch])
        service = torch.Tensor([example['service'] for example in batch])
        noise = torch.Tensor([example['noise'] for example in batch])

        # concatenate the concepts in the same tensor
        concepts = torch.stack([food, ambiance, service, noise], dim=1)

        return {
            'x': {
                'input_ids': input_ids, 
                'token_type_ids': token_type_ids, 
                'attention_mask': attention_mask
            },
            'c': concepts,
            'y': labels
        }
    
def main(): 
    loader = CEBaBDataset('bert-base-uncased', 128)
    train_loader, _, _ = loader.collator()

    for batch in train_loader:
        print(batch['x']['input_ids'])
        print(batch['c'])
        print(batch['y'])
        break

if __name__=="__main__":
    main()