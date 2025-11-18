import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd
from env import DATA_PATH

MAWPS_DIR = f'{DATA_PATH}mawps'

class BERTForMAWPS(nn.Module):
    """
    BERT model for MAWPS dataset.
    Uses BERT encoder with CLS token to predict numerical variables and equation type.
    """
    def __init__(self, num_equations, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, vocab_size=30522):
        super(BERTForMAWPS, self).__init__()
        
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        
        self.bert = BertModel(config)
        
        # Head for predicting 3 numerical variables (N_00, N_01, N_02)
        self.numbers_head = nn.Linear(hidden_size, 3)
        
        # Head for predicting equation type (classification)
        self.equation_head = nn.Linear(hidden_size, num_equations)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for MAWPS prediction task.
        Returns predictions for numbers, equation type, and hidden states.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get the CLS token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Predict numerical variables
        numbers_pred = self.numbers_head(cls_output)  # (batch_size, 3)
        
        # Predict equation type
        equation_logits = self.equation_head(cls_output)  # (batch_size, num_equations)
        
        return numbers_pred, equation_logits, outputs.last_hidden_state


class MAWPSTextDataset(Dataset):
    """
    Dataset for MAWPS sentences with numerical variables and equation types.
    """
    def __init__(self, sentences, numbers, equations, equation_to_idx, tokenizer, max_length=128):
        self.sentences = sentences
        self.numbers = numbers  # List of lists: [[n00, n01, n02], ...]
        self.equations = equations  # List of equation strings
        self.equation_to_idx = equation_to_idx  # Dict mapping equation string to index
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        numbers = self.numbers[idx]  # [N_00, N_01, N_02]
        equation = self.equations[idx]
        equation_idx = self.equation_to_idx[equation]
        
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'numbers': torch.tensor(numbers, dtype=torch.float32),
            'equation_idx': torch.tensor(equation_idx, dtype=torch.long)
        }


class MAWPSBERTPretrainer:
    """
    Pretrains BERT model for MAWPS dataset reconstruction.
    """
    def __init__(
        self,
        use_full_dataset=True,
        pretrained_model_name='bert-base-uncased',
        max_length=128,
        batch_size=32,
        learning_rate=2e-5,
        num_epochs=10,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        model_save_path=None
    ):
        self.use_full_dataset = use_full_dataset
        self.pretrained_model_name = pretrained_model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        
        # Set model save path
        if model_save_path is None:
            self.model_save_path = os.path.join(MAWPS_DIR, 'bert_pretrained')
        else:
            self.model_save_path = model_save_path
            
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        
        # Calculate and save max_length based on dataset
        self._compute_max_length()
        
        # Initialize model
        self.model = None
        
    def _compute_max_length(self):
        """
        Compute the maximum token length in the MAWPS dataset.
        """
        print("Computing maximum token length in MAWPS dataset...")
        
        # Load datasets
        train_df = pd.read_pickle(os.path.join(MAWPS_DIR, 'mawps_train.pkl'))
        
        sentences = train_df['Question'].tolist()
        
        if self.use_full_dataset:
            val_df = pd.read_pickle(os.path.join(MAWPS_DIR, 'mawps_val.pkl'))
            test_df = pd.read_pickle(os.path.join(MAWPS_DIR, 'mawps_test.pkl'))
            sentences.extend(val_df['Question'].tolist())
            sentences.extend(test_df['Question'].tolist())
        
        # Compute max length
        max_tokens = 0
        for sentence in tqdm(sentences, desc="Computing max length"):
            tokens = self.tokenizer.encode(sentence, add_special_tokens=True)
            max_tokens = max(max_tokens, len(tokens))
        
        self.max_length = max_tokens
        print(f"Maximum token length: {self.max_length}")
        
        # Save max_length to file
        with open(os.path.join(self.model_save_path, 'max_length.txt'), 'w') as f:
            f.write(str(self.max_length))
    
    def _prepare_dataloaders(self):
        """
        Prepare dataloaders for training.
        """
        # Load datasets
        train_df = pd.read_pickle(os.path.join(MAWPS_DIR, 'mawps_train.pkl'))
        val_df = pd.read_pickle(os.path.join(MAWPS_DIR, 'mawps_val.pkl'))
        test_df = pd.read_pickle(os.path.join(MAWPS_DIR, 'mawps_test.pkl'))
        
        # Create equation to index mapping from all unique equations
        all_equations = pd.concat([train_df['Standardized_Equation'], 
                                   val_df['Standardized_Equation'], 
                                   test_df['Standardized_Equation']]).unique().tolist()
        self.equation_to_idx = {eq: idx for idx, eq in enumerate(sorted(all_equations))}
        self.idx_to_equation = {idx: eq for eq, idx in self.equation_to_idx.items()}
        self.num_equations = len(self.equation_to_idx)
        
        print(f"\nFound {self.num_equations} unique equations:")
        for eq, idx in sorted(self.equation_to_idx.items(), key=lambda x: x[1]):
            print(f"  {idx}: {eq}")
        
        if self.use_full_dataset:
            # Use all data for training
            all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            all_sentences = all_df['Question'].tolist()
            all_numbers = [[row['N_00'], row['N_01'], row['N_02']] for _, row in all_df.iterrows()]
            all_equations = all_df['Standardized_Equation'].tolist()
            
            train_dataset = MAWPSTextDataset(all_sentences, all_numbers, all_equations, 
                                            self.equation_to_idx, self.tokenizer, self.max_length)
            val_dataset = None  # No separate validation when using full dataset
            test_dataset = None  # No separate test when using full dataset
        else:
            # Use separate splits
            train_sentences = train_df['Question'].tolist()
            train_numbers = [[row['N_00'], row['N_01'], row['N_02']] for _, row in train_df.iterrows()]
            train_equations = train_df['Standardized_Equation'].tolist()
            
            val_sentences = val_df['Question'].tolist()
            val_numbers = [[row['N_00'], row['N_01'], row['N_02']] for _, row in val_df.iterrows()]
            val_equations = val_df['Standardized_Equation'].tolist()
            
            test_sentences = test_df['Question'].tolist()
            test_numbers = [[row['N_00'], row['N_01'], row['N_02']] for _, row in test_df.iterrows()]
            test_equations = test_df['Standardized_Equation'].tolist()
            
            train_dataset = MAWPSTextDataset(train_sentences, train_numbers, train_equations,
                                            self.equation_to_idx, self.tokenizer, self.max_length)
            val_dataset = MAWPSTextDataset(val_sentences, val_numbers, val_equations,
                                          self.equation_to_idx, self.tokenizer, self.max_length)
            test_dataset = MAWPSTextDataset(test_sentences, test_numbers, test_equations,
                                           self.equation_to_idx, self.tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False) if test_dataset else None
        
        return train_loader, val_loader, test_loader
    
    def train(self):
        """
        Train the BERT model for MAWPS prediction tasks.
        """
        print("Starting BERT pretraining for MAWPS dataset...")
        
        # Prepare dataloaders first to get num_equations
        train_loader, val_loader, test_loader = self._prepare_dataloaders()
        
        # Initialize model
        self.model = BERTForMAWPS(
            num_equations=self.num_equations,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            vocab_size=self.tokenizer.vocab_size
        ).to(self.device)
        
        # Loss functions
        mse_criterion = nn.MSELoss()
        ce_criterion = nn.CrossEntropyLoss()
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_metrics = self._train_epoch(train_loader, optimizer, scheduler, mse_criterion, ce_criterion, epoch)
            
            # Display training metrics
            print(f"\nEpoch {epoch+1}/{self.num_epochs} - Training Metrics:")
            print(f"  Total Loss: {train_metrics['total_loss']:.4f}")
            print(f"  MSE Loss: {train_metrics['mse_loss']:.4f}")
            print(f"  CE Loss: {train_metrics['ce_loss']:.4f}")
            print(f"  MAE N_00: {train_metrics['mae_n00']:.4f}")
            print(f"  MAE N_01: {train_metrics['mae_n01']:.4f}")
            print(f"  MAE N_02: {train_metrics['mae_n02']:.4f}")
            print(f"  Equation Accuracy: {train_metrics['eq_accuracy']:.4f}")
            
            # Validation (only if not using full dataset)
            if not self.use_full_dataset and val_loader is not None:
                self.model.eval()
                val_metrics = self._evaluate_epoch(val_loader, mse_criterion, ce_criterion, "Validation")
                
                print(f"\nEpoch {epoch+1}/{self.num_epochs} - Validation Metrics:")
                print(f"  Total Loss: {val_metrics['total_loss']:.4f}")
                print(f"  MSE Loss: {val_metrics['mse_loss']:.4f}")
                print(f"  CE Loss: {val_metrics['ce_loss']:.4f}")
                print(f"  MAE N_00: {val_metrics['mae_n00']:.4f}")
                print(f"  MAE N_01: {val_metrics['mae_n01']:.4f}")
                print(f"  MAE N_02: {val_metrics['mae_n02']:.4f}")
                print(f"  Equation Accuracy: {val_metrics['eq_accuracy']:.4f}")
                
                # Save best model based on validation loss
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    self.save_model()
                    print(f"  → Best model saved! (Val Loss: {best_val_loss:.4f})")
            else:
                # When using full dataset, save model after each epoch
                if epoch == self.num_epochs - 1:
                    self.save_model()
                    print(f"  → Model saved!")
        
        # Test evaluation (only if not using full dataset)
        if not self.use_full_dataset and test_loader is not None:
            print("\nEvaluating on test set...")
            self.model.eval()
            test_metrics = self._evaluate_epoch(test_loader, mse_criterion, ce_criterion, "Test")
            
            print(f"\nTest Metrics:")
            print(f"  Total Loss: {test_metrics['total_loss']:.4f}")
            print(f"  MSE Loss: {test_metrics['mse_loss']:.4f}")
            print(f"  CE Loss: {test_metrics['ce_loss']:.4f}")
            print(f"  MAE N_00: {test_metrics['mae_n00']:.4f}")
            print(f"  MAE N_01: {test_metrics['mae_n01']:.4f}")
            print(f"  MAE N_02: {test_metrics['mae_n02']:.4f}")
            print(f"  Equation Accuracy: {test_metrics['eq_accuracy']:.4f}")
        
        print(f"\nPretraining completed! Model saved at: {self.model_save_path}")
    
    def _train_epoch(self, loader, optimizer, scheduler, mse_criterion, ce_criterion, epoch):
        """
        Train for one epoch and return metrics.
        """
        self.model.train()
        total_loss = 0.0
        total_mse_loss = 0.0
        total_ce_loss = 0.0
        total_mae_n00 = 0.0
        total_mae_n01 = 0.0
        total_mae_n02 = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            numbers_target = batch['numbers'].to(self.device)  # (batch_size, 3)
            equation_target = batch['equation_idx'].to(self.device)  # (batch_size,)
            
            # Forward pass
            numbers_pred, equation_logits, _ = self.model(input_ids, attention_mask)
            
            # Compute losses
            mse_loss = mse_criterion(numbers_pred, numbers_target)
            ce_loss = ce_criterion(equation_logits, equation_target)
            loss = mse_loss + ce_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Compute metrics
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_mse_loss += mse_loss.item() * batch_size
            total_ce_loss += ce_loss.item() * batch_size
            
            # MAE for each variable
            mae = torch.abs(numbers_pred - numbers_target)
            total_mae_n00 += mae[:, 0].sum().item()
            total_mae_n01 += mae[:, 1].sum().item()
            total_mae_n02 += mae[:, 2].sum().item()
            
            # Accuracy for equation prediction
            _, predicted = torch.max(equation_logits, 1)
            total_correct += (predicted == equation_target).sum().item()
            total_samples += batch_size
            
            pbar.set_postfix({
                'loss': loss.item(),
                'mse': mse_loss.item(),
                'ce': ce_loss.item()
            })
        
        return {
            'total_loss': total_loss / total_samples,
            'mse_loss': total_mse_loss / total_samples,
            'ce_loss': total_ce_loss / total_samples,
            'mae_n00': total_mae_n00 / total_samples,
            'mae_n01': total_mae_n01 / total_samples,
            'mae_n02': total_mae_n02 / total_samples,
            'eq_accuracy': total_correct / total_samples
        }
    
    def _evaluate_epoch(self, loader, mse_criterion, ce_criterion, split_name):
        """
        Evaluate for one epoch and return metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_mse_loss = 0.0
        total_ce_loss = 0.0
        total_mae_n00 = 0.0
        total_mae_n01 = 0.0
        total_mae_n02 = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(loader, desc=f"{split_name}")
        
        with torch.no_grad():
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numbers_target = batch['numbers'].to(self.device)
                equation_target = batch['equation_idx'].to(self.device)
                
                # Forward pass
                numbers_pred, equation_logits, _ = self.model(input_ids, attention_mask)
                
                # Compute losses
                mse_loss = mse_criterion(numbers_pred, numbers_target)
                ce_loss = ce_criterion(equation_logits, equation_target)
                loss = mse_loss + ce_loss
                
                # Compute metrics
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_mse_loss += mse_loss.item() * batch_size
                total_ce_loss += ce_loss.item() * batch_size
                
                # MAE for each variable
                mae = torch.abs(numbers_pred - numbers_target)
                total_mae_n00 += mae[:, 0].sum().item()
                total_mae_n01 += mae[:, 1].sum().item()
                total_mae_n02 += mae[:, 2].sum().item()
                
                # Accuracy for equation prediction
                _, predicted = torch.max(equation_logits, 1)
                total_correct += (predicted == equation_target).sum().item()
                total_samples += batch_size
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': total_correct / total_samples
                })
        
        return {
            'total_loss': total_loss / total_samples,
            'mse_loss': total_mse_loss / total_samples,
            'ce_loss': total_ce_loss / total_samples,
            'mae_n00': total_mae_n00 / total_samples,
            'mae_n01': total_mae_n01 / total_samples,
            'mae_n02': total_mae_n02 / total_samples,
            'eq_accuracy': total_correct / total_samples
        }
    
    def save_model(self):
        """
        Save the pretrained model, tokenizer, and equation mappings.
        """
        # Save model state
        model_path = os.path.join(self.model_save_path, 'bert_model.pt')
        torch.save(self.model.state_dict(), model_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.model_save_path)
        
        # Save max_length
        with open(os.path.join(self.model_save_path, 'max_length.txt'), 'w') as f:
            f.write(str(self.max_length))
        
        # Save equation mappings
        import json
        with open(os.path.join(self.model_save_path, 'equation_mappings.json'), 'w') as f:
            json.dump({
                'equation_to_idx': self.equation_to_idx,
                'idx_to_equation': self.idx_to_equation,
                'num_equations': self.num_equations
            }, f, indent=2)
    
    @staticmethod
    def load_pretrained_model(model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Load a pretrained BERT model.
        """
        if model_path is None:
            model_path = os.path.join(MAWPS_DIR, 'bert_pretrained')
        
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_path)
        
        # Load max_length
        with open(os.path.join(model_path, 'max_length.txt'), 'r') as f:
            max_length = int(f.read().strip())
        
        # Load equation mappings
        import json
        with open(os.path.join(model_path, 'equation_mappings.json'), 'r') as f:
            mappings = json.load(f)
            num_equations = mappings['num_equations']
        
        # Initialize model
        model = BERTForMAWPS(
            num_equations=num_equations,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            vocab_size=tokenizer.vocab_size
        ).to(device)
        
        # Load model weights
        model.load_state_dict(torch.load(os.path.join(model_path, 'bert_model.pt'), map_location=device))
        model.eval()
        
        return model, tokenizer, max_length
    
    @staticmethod
    def model_exists(model_path=None):
        """
        Check if a pretrained model exists.
        """
        if model_path is None:
            model_path = os.path.join(MAWPS_DIR, 'bert_pretrained')
        
        model_file = os.path.join(model_path, 'bert_model.pt')
        return os.path.exists(model_file)


def extract_bert_embeddings(model, tokenizer, sentences, max_length, device='cuda', batch_size=32):
    """
    Extract BERT CLS embeddings for a list of sentences.
    Returns embeddings of shape (num_sentences, 768).
    
    Args:
        model: Pretrained BERTForMAWPS model
        tokenizer: BERT tokenizer
        sentences: List of sentences
        max_length: Maximum sequence length
        device: Device to run on
        batch_size: Batch size for processing
    
    Returns:
        embeddings: Tensor of shape (num_sentences, 768) - CLS token embeddings only
    """
    model.eval()
    all_embeddings = []
    
    # Create a simple dataset without numbers/equations for embedding extraction
    from torch.utils.data import TensorDataset
    
    # Tokenize all sentences
    encodings = tokenizer(
        sentences,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Get hidden states
            _, _, hidden_states = model(input_ids, attention_mask)
            
            # Extract only CLS token embedding (first token)
            cls_embeddings = hidden_states[:, 0, :]  # (batch_size, 768)
            all_embeddings.append(cls_embeddings.cpu())
    
    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings
