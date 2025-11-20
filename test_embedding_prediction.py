"""
Test script to evaluate whether concepts and equations can be predicted from text embeddings.

This script:
1. Loads the MAWPS dataset
2. Extracts embeddings using the pre-trained text backbone from config
3. Trains separate MLPs to predict each concept (N_00, N_01, N_02) using MSE loss
4. Trains an MLP to predict the equation type using Cross Entropy loss
5. Reports accuracy and loss metrics
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import sympy as sp

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from env import DATA_PATH


class ConceptMLP(nn.Module):
    """MLP to predict a single continuous concept value."""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


class EquationMLP(nn.Module):
    """MLP to predict the equation type (classification)."""
    def __init__(self, input_dim, num_equations, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_equations)
        )
    
    def forward(self, x):
        return self.network(x)


class ConceptEquationToTargetMLP(nn.Module):
    """MLP to predict target from concepts (3 floats) and equation index (1 integer)."""
    def __init__(self, num_equations, hidden_dim=256, embedding_dim=16):
        super().__init__()
        self.equation_embedding = nn.Embedding(num_equations, embedding_dim)
        
        # Input: 3 concepts + equation embedding
        input_dim = 3 + embedding_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, concepts, equation_indices):
        """
        Args:
            concepts: (batch_size, 3) - the three concept values
            equation_indices: (batch_size,) - equation indices
        """
        eq_emb = self.equation_embedding(equation_indices)
        x = torch.cat([concepts, eq_emb], dim=1)
        return self.network(x).squeeze(-1)


def load_mawps_data(seed=42):
    """Load MAWPS dataset splits for a specific seed."""
    mawps_dir = os.path.join(DATA_PATH, 'mawps')
    
    train_df = pd.read_pickle(os.path.join(mawps_dir, f'mawps_train_seed{seed}.pkl'))
    val_df = pd.read_pickle(os.path.join(mawps_dir, f'mawps_val_seed{seed}.pkl'))
    test_df = pd.read_pickle(os.path.join(mawps_dir, f'mawps_test_seed{seed}.pkl'))
    
    return train_df, val_df, test_df


def extract_embeddings(texts, model, tokenizer, device, batch_size=32, max_length=128):
    """Extract embeddings from text using the pre-trained model."""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Check if model expects token_type_ids
            try:
                if 'token_type_ids' in encoded:
                    token_type_ids = encoded['token_type_ids'].to(device)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
            except:
                # Fallback without token_type_ids
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Flatten all hidden states
            embeddings = outputs.last_hidden_state
            embeddings = embeddings.flatten(1).float()
            
            all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0)


def prepare_data(df, embeddings, equation_to_idx):
    """Prepare tensors for training."""
    concepts = torch.tensor([
        [row['N_00'], row['N_01'], row['N_02']] 
        for _, row in df.iterrows()
    ], dtype=torch.float32)
    
    equations = torch.tensor([
        equation_to_idx[row['Equation']] 
        for _, row in df.iterrows()
    ], dtype=torch.long)
    
    return embeddings, concepts, equations


def train_concept_predictor(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    """Train a concept predictor."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience = 100
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for embeddings, concepts in train_loader:
            embeddings, concepts = embeddings.to(device), concepts.to(device)
            
            optimizer.zero_grad()
            predictions = model(embeddings)
            loss = criterion(predictions, concepts)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for embeddings, concepts in val_loader:
                embeddings, concepts = embeddings.to(device), concepts.to(device)
                predictions = model(embeddings)
                loss = criterion(predictions, concepts)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return best_val_loss


def train_equation_predictor(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    """Train an equation predictor."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for embeddings, equations in train_loader:
            embeddings, equations = embeddings.to(device), equations.to(device)
            
            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, equations)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == equations).sum().item()
            train_total += equations.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for embeddings, equations in val_loader:
                embeddings, equations = embeddings.to(device), equations.to(device)
                logits = model(embeddings)
                loss = criterion(logits, equations)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == equations).sum().item()
                val_total += equations.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Early stopping based on accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return best_val_acc


def evaluate_concept_predictor(model, test_loader, device):
    """Evaluate concept predictor on test set."""
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for embeddings, concepts in test_loader:
            embeddings, concepts = embeddings.to(device), concepts.to(device)
            predictions = model(embeddings)
            loss = criterion(predictions, concepts)
            test_loss += loss.item()
            
            all_predictions.append(predictions.cpu())
            all_targets.append(concepts.cpu())
    
    test_loss /= len(test_loader)
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # Calculate MAE as well
    mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
    
    return test_loss, mae


def evaluate_equation_predictor(model, test_loader, device):
    """Evaluate equation predictor on test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for embeddings, equations in test_loader:
            embeddings, equations = embeddings.to(device), equations.to(device)
            logits = model(embeddings)
            loss = criterion(logits, equations)
            test_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            test_correct += (predicted == equations).sum().item()
            test_total += equations.size(0)
    
    test_loss /= len(test_loader)
    test_acc = test_correct / test_total
    
    return test_loss, test_acc


def train_concept_equation_to_target_predictor(model, train_loader, val_loader, device, epochs=50, lr=1e-3):
    """Train the concept+equation to target predictor."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for concepts, equation_indices, targets in train_loader:
            concepts = concepts.to(device)
            equation_indices = equation_indices.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(concepts, equation_indices)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for concepts, equation_indices, targets in val_loader:
                concepts = concepts.to(device)
                equation_indices = equation_indices.to(device)
                targets = targets.to(device)
                predictions = model(concepts, equation_indices)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return best_val_loss


def evaluate_concept_equation_to_target_predictor(model, test_loader, device):
    """Evaluate the concept+equation to target predictor."""
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for concepts, equation_indices, targets in test_loader:
            concepts = concepts.to(device)
            equation_indices = equation_indices.to(device)
            targets = targets.to(device)
            predictions = model(concepts, equation_indices)
            loss = criterion(predictions, targets)
            test_loss += loss.item()
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    test_loss /= len(test_loader)
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # Calculate MAE and R²
    mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
    
    ss_res = torch.sum((all_targets - all_predictions) ** 2).item()
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2).item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return test_loss, mae, r2


def execute_equation(equation_str, n_00, n_01, n_02):
    """
    Execute an equation with given numbers using sympy.
    
    Args:
        equation_str: Equation string like 'N_00+N_01+N_02', 'N_00+N_01-N_02', etc.
        n_00, n_01, n_02: The three concept values (numbers)
    
    Returns:
        Result of the equation execution
    """
    try:
        # Parse the equation
        if '=' in equation_str:
            left, right = equation_str.split('=')
            expr_str = right.strip()
        else:
            expr_str = equation_str.strip()
        
        # Create sympy symbols
        N_00, N_01, N_02 = sp.symbols('N_00 N_01 N_02')
        
        # Parse the expression
        expr = sp.sympify(expr_str)
        
        # Substitute the values: N_00->n_00, N_01->n_01, N_02->n_02
        result = expr.subs([(N_00, n_00), (N_01, n_01), (N_02, n_02)])
        
        # Convert to float
        return float(result)
    except Exception as e:
        print(f"Error executing equation {equation_str}: {e}")
        return 0.0


def evaluate_combined_model(concept_models, equation_model, embeddings, df, 
                            equation_to_idx, idx_to_equation, device, split_name="Test"):
    """
    Evaluate the combined model that predicts concepts and equations, 
    then executes the predicted equation with predicted numbers.
    
    Args:
        concept_models: List of trained concept predictor models
        equation_model: Trained equation predictor model
        embeddings: Text embeddings
        df: DataFrame with ground truth data
        equation_to_idx: Mapping from equation strings to indices
        idx_to_equation: Mapping from indices to equation strings
        device: torch device
        split_name: Name of the split (for logging)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Set all models to eval mode
    for model in concept_models:
        model.eval()
    equation_model.eval()
    
    all_predictions = []
    all_targets = []
    all_concept_predictions = []
    all_equation_predictions = []
    
    with torch.no_grad():
        embeddings = embeddings.to(device)
        
        # Predict concepts
        predicted_concepts = []
        for concept_model in concept_models:
            pred = concept_model(embeddings)
            predicted_concepts.append(pred.cpu().numpy())
        
        predicted_concepts = torch.tensor(predicted_concepts).T  # Shape: (num_samples, 3)
        
        # Predict equations
        equation_logits = equation_model(embeddings)
        predicted_equation_indices = torch.argmax(equation_logits, dim=1).cpu().numpy()
        
        # Execute predicted equations with predicted concepts
        for i in range(len(df)):
            eq_idx = predicted_equation_indices[i]
            equation_str = idx_to_equation[eq_idx]
            
            n_00_pred = predicted_concepts[i, 0].item()
            n_01_pred = predicted_concepts[i, 1].item()
            n_02_pred = predicted_concepts[i, 2].item()
            
            try:
                result = execute_equation(equation_str, n_00_pred, n_01_pred, n_02_pred)
            except Exception as e:
                print(f"Error executing equation {equation_str}: {e}")
                result = 0.0
            
            all_predictions.append(result)
            all_targets.append(df.iloc[i]['Answer'])
            all_concept_predictions.append([n_00_pred, n_01_pred, n_02_pred])
            all_equation_predictions.append(equation_str)
    
    # Convert to arrays
    all_predictions = torch.tensor(all_predictions)
    all_targets = torch.tensor(all_targets)
    
    # Calculate metrics
    mse = F.mse_loss(all_predictions, all_targets).item()
    mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
    
    # Calculate R^2 score
    ss_res = torch.sum((all_targets - all_predictions) ** 2).item()
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2).item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Equation accuracy
    true_equation_indices = [equation_to_idx[row['Equation']] 
                            for _, row in df.iterrows()]
    equation_accuracy = sum(predicted_equation_indices[i] == true_equation_indices[i] 
                           for i in range(len(df))) / len(df)
    
    results = {
        'split': split_name,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'equation_accuracy': equation_accuracy,
        'predictions': all_predictions.numpy(),
        'targets': all_targets.numpy(),
        'concept_predictions': all_concept_predictions,
        'equation_predictions': all_equation_predictions
    }
    
    return results


@hydra.main(version_base=None, config_path="conf", config_name="common")
def main(cfg: DictConfig):
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("\n" + "="*80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Get seed from config
    seed = cfg.seed
    print(f"Using seed: {seed}\n")
    
    # Load text backbone name from config
    text_backbone_name = cfg.text_backbone_name
    print(f"Using text backbone: {text_backbone_name}\n")
    
    # Load MAWPS data
    print("Loading MAWPS dataset...")
    train_df, val_df, test_df = load_mawps_data(seed=seed)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Get unique equations from all splits to create mapping
    all_equations = set(train_df['Equation'].unique()) | \
                    set(val_df['Equation'].unique()) | \
                    set(test_df['Equation'].unique())
    unique_equations = sorted(all_equations)
    equation_to_idx = {eq: idx for idx, eq in enumerate(unique_equations)}
    idx_to_equation = {idx: eq for eq, idx in equation_to_idx.items()}
    num_equations = len(unique_equations)
    print(f"\nNumber of unique equations: {num_equations}")
    for idx, eq in idx_to_equation.items():
        print(f"  {idx}: {eq}")
    
    # Load pre-trained model and tokenizer
    print(f"\nLoading pre-trained model: {text_backbone_name}")
    tokenizer = AutoTokenizer.from_pretrained(text_backbone_name)
    model = AutoModel.from_pretrained(text_backbone_name, torch_dtype=torch.bfloat16, use_safetensors=True)
    model = model.to(device)
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    train_embeddings = extract_embeddings(
        train_df['Question'].tolist(), model, tokenizer, device
    )
    val_embeddings = extract_embeddings(
        val_df['Question'].tolist(), model, tokenizer, device
    )
    test_embeddings = extract_embeddings(
        test_df['Question'].tolist(), model, tokenizer, device
    )
    
    embedding_dim = train_embeddings.shape[1]
    print(f"Embedding dimension: {embedding_dim}")
    
    # Free up memory
    del model
    torch.cuda.empty_cache()
    
    # Prepare data
    print("\nPreparing data for training...")
    train_emb, train_concepts, train_equations = prepare_data(
        train_df, train_embeddings, equation_to_idx
    )
    val_emb, val_concepts, val_equations = prepare_data(
        val_df, val_embeddings, equation_to_idx
    )
    test_emb, test_concepts, test_equations = prepare_data(
        test_df, test_embeddings, equation_to_idx
    )
    
    batch_size = 128
    
    # Train concept predictors (one for each concept)
    print("\n" + "="*80)
    print("TRAINING CONCEPT PREDICTORS")
    print("="*80)
    
    concept_models = []
    concept_names = ['N_00', 'N_01', 'N_02']
    
    for i, concept_name in enumerate(concept_names):
        print(f"\n--- Training predictor for {concept_name} ---")
        
        # Create data loaders for this concept
        train_dataset = TensorDataset(train_emb, train_concepts[:, i])
        val_dataset = TensorDataset(val_emb, val_concepts[:, i])
        test_dataset = TensorDataset(test_emb, test_concepts[:, i])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Create and train model
        concept_model = ConceptMLP(embedding_dim).to(device)
        best_val_loss = train_concept_predictor(
            concept_model, train_loader, val_loader, device, epochs=50, lr=1e-3
        )
        
        # Evaluate on test set
        test_loss, test_mae = evaluate_concept_predictor(concept_model, test_loader, device)
        print(f"\nTest Results for {concept_name}:")
        print(f"  MSE: {test_loss:.4f}")
        print(f"  MAE: {test_mae:.4f}")
        
        concept_models.append(concept_model)
    
    # Train equation predictor
    print("\n" + "="*80)
    print("TRAINING EQUATION PREDICTOR")
    print("="*80)
    
    train_dataset = TensorDataset(train_emb, train_equations)
    val_dataset = TensorDataset(val_emb, val_equations)
    test_dataset = TensorDataset(test_emb, test_equations)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    equation_model = EquationMLP(embedding_dim, num_equations).to(device)
    best_val_acc = train_equation_predictor(
        equation_model, train_loader, val_loader, device, epochs=50, lr=1e-3
    )
    
    # Evaluate on test set
    test_loss, test_acc = evaluate_equation_predictor(equation_model, test_loader, device)
    print(f"\nTest Results for Equation Prediction:")
    print(f"  Cross Entropy Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Evaluate combined model on all splits
    print("\n" + "="*80)
    print("EVALUATING COMBINED MODEL (Concepts + Equation Execution)")
    print("="*80)
    
    # Train split
    print("\n--- Train Split ---")
    train_results = evaluate_combined_model(
        concept_models, equation_model, train_emb, train_df,
        equation_to_idx, idx_to_equation, device, split_name="Train"
    )
    print(f"MSE: {train_results['mse']:.4f}")
    print(f"MAE: {train_results['mae']:.4f}")
    print(f"R²: {train_results['r2']:.4f}")
    print(f"Equation Accuracy: {train_results['equation_accuracy']:.4f} ({train_results['equation_accuracy']*100:.2f}%)")
    
    # Validation split
    print("\n--- Validation Split ---")
    val_results = evaluate_combined_model(
        concept_models, equation_model, val_emb, val_df,
        equation_to_idx, idx_to_equation, device, split_name="Validation"
    )
    print(f"MSE: {val_results['mse']:.4f}")
    print(f"MAE: {val_results['mae']:.4f}")
    print(f"R²: {val_results['r2']:.4f}")
    print(f"Equation Accuracy: {val_results['equation_accuracy']:.4f} ({val_results['equation_accuracy']*100:.2f}%)")
    
    # Test split
    print("\n--- Test Split ---")
    test_results = evaluate_combined_model(
        concept_models, equation_model, test_emb, test_df,
        equation_to_idx, idx_to_equation, device, split_name="Test"
    )
    print(f"MSE: {test_results['mse']:.4f}")
    print(f"MAE: {test_results['mae']:.4f}")
    print(f"R²: {test_results['r2']:.4f}")
    print(f"Equation Accuracy: {test_results['equation_accuracy']:.4f} ({test_results['equation_accuracy']*100:.2f}%)")
    
    # Sanity check: Execute predicted equations with ground truth numbers
    print("\n" + "="*80)
    print("SANITY CHECK: Predicted Equations + Ground Truth Concepts")
    print("="*80)
    
    def sanity_check_predictions(equation_model, embeddings, df, equation_to_idx, idx_to_equation, device, split_name="Test"):
        """Execute predicted equations using ground truth concept values."""
        equation_model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            embeddings = embeddings.to(device)
            equation_logits = equation_model(embeddings)
            predicted_equation_indices = torch.argmax(equation_logits, dim=1).cpu().numpy()
            
            for i in range(len(df)):
                eq_idx = predicted_equation_indices[i]
                equation_str = idx_to_equation[eq_idx]
                
                # Use ground truth concepts
                row = df.iloc[i]
                n_00_true = row['N_00']
                n_01_true = row['N_01']
                n_02_true = row['N_02']
                
                result = execute_equation(equation_str, n_00_true, n_01_true, n_02_true)
                all_predictions.append(result)
                all_targets.append(row['Answer'])
        
        all_predictions = torch.tensor(all_predictions)
        all_targets = torch.tensor(all_targets)
        
        mse = F.mse_loss(all_predictions, all_targets).item()
        mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
        
        ss_res = torch.sum((all_targets - all_predictions) ** 2).item()
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        true_equation_indices = [equation_to_idx[row['Equation']] 
                                for _, row in df.iterrows()]
        equation_accuracy = sum(predicted_equation_indices[i] == true_equation_indices[i] 
                               for i in range(len(df))) / len(df)
        
        return {
            'split': split_name,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'equation_accuracy': equation_accuracy
        }
    
    # Train split sanity check
    print("\n--- Train Split (Sanity Check) ---")
    train_sanity = sanity_check_predictions(
        equation_model, train_emb, train_df, equation_to_idx, idx_to_equation, device, "Train"
    )
    print(f"MSE: {train_sanity['mse']:.4f}")
    print(f"MAE: {train_sanity['mae']:.4f}")
    print(f"R²: {train_sanity['r2']:.4f}")
    print(f"Equation Accuracy: {train_sanity['equation_accuracy']:.4f} ({train_sanity['equation_accuracy']*100:.2f}%)")
    
    # Validation split sanity check
    print("\n--- Validation Split (Sanity Check) ---")
    val_sanity = sanity_check_predictions(
        equation_model, val_emb, val_df, equation_to_idx, idx_to_equation, device, "Validation"
    )
    print(f"MSE: {val_sanity['mse']:.4f}")
    print(f"MAE: {val_sanity['mae']:.4f}")
    print(f"R²: {val_sanity['r2']:.4f}")
    print(f"Equation Accuracy: {val_sanity['equation_accuracy']:.4f} ({val_sanity['equation_accuracy']*100:.2f}%)")
    
    # Test split sanity check
    print("\n--- Test Split (Sanity Check) ---")
    test_sanity = sanity_check_predictions(
        equation_model, test_emb, test_df, equation_to_idx, idx_to_equation, device, "Test"
    )
    print(f"MSE: {test_sanity['mse']:.4f}")
    print(f"MAE: {test_sanity['mae']:.4f}")
    print(f"R²: {test_sanity['r2']:.4f}")
    print(f"Equation Accuracy: {test_sanity['equation_accuracy']:.4f} ({test_sanity['equation_accuracy']*100:.2f}%)")
    
    # Perfect reconstruction: Execute ground truth equations with ground truth concepts
    print("\n" + "="*80)
    print("PERFECT RECONSTRUCTION: Ground Truth Equations + Ground Truth Concepts")
    print("="*80)
    
    def perfect_reconstruction_check(df, split_name="Test"):
        """Execute ground truth equations with ground truth concept values."""
        all_predictions = []
        all_targets = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            equation_str = row['Equation']
            
            # Use ground truth concepts
            n_00_true = row['N_00']
            n_01_true = row['N_01']
            n_02_true = row['N_02']
            
            result = execute_equation(equation_str, n_00_true, n_01_true, n_02_true)
            all_predictions.append(result)
            all_targets.append(row['Answer'])
        
        all_predictions = torch.tensor(all_predictions)
        all_targets = torch.tensor(all_targets)
        
        mse = F.mse_loss(all_predictions, all_targets).item()
        mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
        
        ss_res = torch.sum((all_targets - all_predictions) ** 2).item()
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2).item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'split': split_name,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
    
    # Train split perfect reconstruction
    print("\n--- Train Split (Perfect Reconstruction) ---")
    train_perfect = perfect_reconstruction_check(train_df, "Train")
    print(f"MSE: {train_perfect['mse']:.4f}")
    print(f"MAE: {train_perfect['mae']:.4f}")
    print(f"R²: {train_perfect['r2']:.4f}")
    
    # Validation split perfect reconstruction
    print("\n--- Validation Split (Perfect Reconstruction) ---")
    val_perfect = perfect_reconstruction_check(val_df, "Validation")
    print(f"MSE: {val_perfect['mse']:.4f}")
    print(f"MAE: {val_perfect['mae']:.4f}")
    print(f"R²: {val_perfect['r2']:.4f}")
    
    # Test split perfect reconstruction
    print("\n--- Test Split (Perfect Reconstruction) ---")
    test_perfect = perfect_reconstruction_check(test_df, "Test")
    print(f"MSE: {test_perfect['mse']:.4f}")
    print(f"MAE: {test_perfect['mae']:.4f}")
    print(f"R²: {test_perfect['r2']:.4f}")
    
    # Intervention analysis: Replace predicted concepts with ground truth at varying rates
    print("\n" + "="*80)
    print("INTERVENTION ANALYSIS: Effect of Replacing Predicted Concepts with Ground Truth")
    print("="*80)
    
    def intervention_analysis(concept_models, equation_model, embeddings, df, 
                             equation_to_idx, idx_to_equation, device, 
                             intervention_probs=None, split_name="Test"):
        """
        Evaluate model performance with varying levels of concept intervention.
        
        Args:
            concept_models: List of trained concept predictor models
            equation_model: Trained equation predictor model
            embeddings: Text embeddings
            df: DataFrame with ground truth data
            equation_to_idx: Mapping from equation strings to indices
            idx_to_equation: Mapping from indices to equation strings
            device: torch device
            intervention_probs: List of intervention probabilities to test
            split_name: Name of the split (for logging)
        
        Returns:
            Dictionary with intervention probabilities and corresponding MAE values
        """
        if intervention_probs is None:
            intervention_probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        import numpy as np
        
        results = {
            'intervention_probs': [],
            'mae_values': [],
            'num_intervened': []
        }
        
        # Set all models to eval mode
        for model in concept_models:
            model.eval()
        equation_model.eval()
        
        # Get predicted concepts and equations once
        with torch.no_grad():
            embeddings_gpu = embeddings.to(device)
            
            # Predict concepts
            predicted_concepts = []
            for concept_model in concept_models:
                pred = concept_model(embeddings_gpu)
                predicted_concepts.append(pred.cpu().numpy())
            predicted_concepts = torch.tensor(predicted_concepts).T  # Shape: (num_samples, 3)
            
            # Predict equations
            equation_logits = equation_model(embeddings_gpu)
            predicted_equation_indices = torch.argmax(equation_logits, dim=1).cpu().numpy()
        
        # Get ground truth concepts
        gt_concepts = torch.tensor([
            [row['N_00'], row['N_01'], row['N_02']] 
            for _, row in df.iterrows()
        ], dtype=torch.float32)
        
        # Test each intervention probability
        for intervention_prob in intervention_probs:
            # Determine which samples to intervene on
            np.random.seed(42)  # For reproducibility
            num_samples = len(df)
            intervene_mask = np.random.rand(num_samples) < intervention_prob
            num_intervened = intervene_mask.sum()
            
            # Create concepts with intervention
            concepts_with_intervention = predicted_concepts.clone()
            concepts_with_intervention[intervene_mask] = gt_concepts[intervene_mask]
            
            # Execute equations
            all_predictions = []
            all_targets = []
            
            for i in range(len(df)):
                eq_idx = predicted_equation_indices[i]
                equation_str = idx_to_equation[eq_idx]
                
                n_00 = concepts_with_intervention[i, 0].item()
                n_01 = concepts_with_intervention[i, 1].item()
                n_02 = concepts_with_intervention[i, 2].item()
                
                result = execute_equation(equation_str, n_00, n_01, n_02)
                all_predictions.append(result)
                all_targets.append(df.iloc[i]['Answer'])
            
            # Calculate MAE
            all_predictions = torch.tensor(all_predictions)
            all_targets = torch.tensor(all_targets)
            mae = torch.mean(torch.abs(all_predictions - all_targets)).item()
            
            results['intervention_probs'].append(intervention_prob)
            results['mae_values'].append(mae)
            results['num_intervened'].append(num_intervened)
            
            print(f"  Intervention {intervention_prob:.1f} ({num_intervened}/{num_samples} samples): MAE = {mae:.4f}")
        
        return results
    
    # Run intervention analysis on test set
    print(f"\n--- Test Set Intervention Analysis ---")
    test_intervention = intervention_analysis(
        concept_models, equation_model, test_emb, test_df,
        equation_to_idx, idx_to_equation, device, split_name="Test"
    )
    
    # Create visualization
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(test_intervention['num_intervened'], 
            test_intervention['mae_values'], 
            marker='o', linewidth=2, markersize=8, color='#2E86AB')
    
    ax.set_xlabel('Number of Intervened Samples', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax.set_title('Effect of Concept Intervention on Model Performance\n(Test Set)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add intervention probability labels on top x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([test_intervention['num_intervened'][i] 
                    for i in range(0, len(test_intervention['num_intervened']), 2)])
    ax2.set_xticklabels([f"{test_intervention['intervention_probs'][i]:.1f}" 
                         for i in range(0, len(test_intervention['intervention_probs']), 2)])
    ax2.set_xlabel('Intervention Probability', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results/figs', exist_ok=True)
    fig_path = f'results/figs/intervention_analysis_seed{seed}.pdf'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved intervention analysis plot to: {fig_path}")
    plt.close()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Seed: {seed}")
    print(f"Text Backbone: {text_backbone_name}")
    print(f"Embedding Dimension: {embedding_dim}")
    print(f"\nConcept Prediction (MSE/MAE on test set):")
    for i, concept_name in enumerate(concept_names):
        test_dataset_i = TensorDataset(test_emb, test_concepts[:, i])
        test_loader_i = DataLoader(test_dataset_i, batch_size=batch_size, shuffle=False)
        test_loss, test_mae = evaluate_concept_predictor(concept_models[i], test_loader_i, device)
        print(f"  {concept_name}: MSE={test_loss:.4f}, MAE={test_mae:.4f}")
    
    print(f"\nEquation Prediction:")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    print(f"\nCombined Model (End-to-End Target Prediction):")
    print(f"  Train  - MSE: {train_results['mse']:.4f}, MAE: {train_results['mae']:.4f}, R²: {train_results['r2']:.4f}")
    print(f"  Val    - MSE: {val_results['mse']:.4f}, MAE: {val_results['mae']:.4f}, R²: {val_results['r2']:.4f}")
    print(f"  Test   - MSE: {test_results['mse']:.4f}, MAE: {test_results['mae']:.4f}, R²: {test_results['r2']:.4f}")
    
    print(f"\nSanity Check (Predicted Equations + Ground Truth Concepts):")
    print(f"  Train  - MSE: {train_sanity['mse']:.4f}, MAE: {train_sanity['mae']:.4f}, R²: {train_sanity['r2']:.4f}")
    print(f"  Val    - MSE: {val_sanity['mse']:.4f}, MAE: {val_sanity['mae']:.4f}, R²: {val_sanity['r2']:.4f}")
    print(f"  Test   - MSE: {test_sanity['mse']:.4f}, MAE: {test_sanity['mae']:.4f}, R²: {test_sanity['r2']:.4f}")
    
    print(f"\nPerfect Reconstruction (Ground Truth Equations + Ground Truth Concepts):")
    print(f"  Train  - MSE: {train_perfect['mse']:.4f}, MAE: {train_perfect['mae']:.4f}, R²: {train_perfect['r2']:.4f}")
    print(f"  Val    - MSE: {val_perfect['mse']:.4f}, MAE: {val_perfect['mae']:.4f}, R²: {val_perfect['r2']:.4f}")
    print(f"  Test   - MSE: {test_perfect['mse']:.4f}, MAE: {test_perfect['mae']:.4f}, R²: {test_perfect['r2']:.4f}")
    
    # Train MLP to predict target from concepts + equation index
    print("\n" + "="*80)
    print("SANITY CHECK: MLP Predicting Target from Concepts + Equation Index")
    print("="*80)
    
    # Prepare datasets
    train_targets = torch.tensor(train_df['Answer'].values, dtype=torch.float32)
    val_targets = torch.tensor(val_df['Answer'].values, dtype=torch.float32)
    test_targets = torch.tensor(test_df['Answer'].values, dtype=torch.float32)
    
    train_dataset_ce2t = TensorDataset(train_concepts, train_equations, train_targets)
    val_dataset_ce2t = TensorDataset(val_concepts, val_equations, val_targets)
    test_dataset_ce2t = TensorDataset(test_concepts, test_equations, test_targets)
    
    train_loader_ce2t = DataLoader(train_dataset_ce2t, batch_size=batch_size, shuffle=True)
    val_loader_ce2t = DataLoader(val_dataset_ce2t, batch_size=batch_size, shuffle=False)
    test_loader_ce2t = DataLoader(test_dataset_ce2t, batch_size=batch_size, shuffle=False)
    
    # Create and train model
    print("\nTraining MLP to predict target from ground truth concepts + equation index...")
    ce2t_model = ConceptEquationToTargetMLP(num_equations=num_equations).to(device)
    best_val_loss_ce2t = train_concept_equation_to_target_predictor(
        ce2t_model, train_loader_ce2t, val_loader_ce2t, device, epochs=50, lr=1e-3
    )
    
    # Evaluate on all splits
    print("\n--- Train Split ---")
    train_loss_ce2t, train_mae_ce2t, train_r2_ce2t = evaluate_concept_equation_to_target_predictor(
        ce2t_model, train_loader_ce2t, device
    )
    print(f"MSE: {train_loss_ce2t:.4f}")
    print(f"MAE: {train_mae_ce2t:.4f}")
    print(f"R²: {train_r2_ce2t:.4f}")
    
    print("\n--- Validation Split ---")
    val_loss_ce2t, val_mae_ce2t, val_r2_ce2t = evaluate_concept_equation_to_target_predictor(
        ce2t_model, val_loader_ce2t, device
    )
    print(f"MSE: {val_loss_ce2t:.4f}")
    print(f"MAE: {val_mae_ce2t:.4f}")
    print(f"R²: {val_r2_ce2t:.4f}")
    
    print("\n--- Test Split ---")
    test_loss_ce2t, test_mae_ce2t, test_r2_ce2t = evaluate_concept_equation_to_target_predictor(
        ce2t_model, test_loader_ce2t, device
    )
    print(f"MSE: {test_loss_ce2t:.4f}")
    print(f"MAE: {test_mae_ce2t:.4f}")
    print(f"R²: {test_r2_ce2t:.4f}")
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Seed: {seed}")
    print(f"Text Backbone: {text_backbone_name}")
    print(f"Embedding Dimension: {embedding_dim}")
    print(f"\nConcept Prediction (MSE/MAE on test set):")
    for i, concept_name in enumerate(concept_names):
        test_dataset_i = TensorDataset(test_emb, test_concepts[:, i])
        test_loader_i = DataLoader(test_dataset_i, batch_size=batch_size, shuffle=False)
        test_loss, test_mae = evaluate_concept_predictor(concept_models[i], test_loader_i, device)
        print(f"  {concept_name}: MSE={test_loss:.4f}, MAE={test_mae:.4f}")
    
    print(f"\nEquation Prediction:")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    print(f"\nCombined Model (End-to-End Target Prediction):")
    print(f"  Train  - MSE: {train_results['mse']:.4f}, MAE: {train_results['mae']:.4f}, R²: {train_results['r2']:.4f}")
    print(f"  Val    - MSE: {val_results['mse']:.4f}, MAE: {val_results['mae']:.4f}, R²: {val_results['r2']:.4f}")
    print(f"  Test   - MSE: {test_results['mse']:.4f}, MAE: {test_results['mae']:.4f}, R²: {test_results['r2']:.4f}")
    
    print(f"\nSanity Check (Predicted Equations + Ground Truth Concepts):")
    print(f"  Train  - MSE: {train_sanity['mse']:.4f}, MAE: {train_sanity['mae']:.4f}, R²: {train_sanity['r2']:.4f}")
    print(f"  Val    - MSE: {val_sanity['mse']:.4f}, MAE: {val_sanity['mae']:.4f}, R²: {val_sanity['r2']:.4f}")
    print(f"  Test   - MSE: {test_sanity['mse']:.4f}, MAE: {test_sanity['mae']:.4f}, R²: {test_sanity['r2']:.4f}")
    
    print(f"\nPerfect Reconstruction (Ground Truth Equations + Ground Truth Concepts):")
    print(f"  Train  - MSE: {train_perfect['mse']:.4f}, MAE: {train_perfect['mae']:.4f}, R²: {train_perfect['r2']:.4f}")
    print(f"  Val    - MSE: {val_perfect['mse']:.4f}, MAE: {val_perfect['mae']:.4f}, R²: {val_perfect['r2']:.4f}")
    print(f"  Test   - MSE: {test_perfect['mse']:.4f}, MAE: {test_perfect['mae']:.4f}, R²: {test_perfect['r2']:.4f}")
    
    print(f"\nMLP Sanity Check (Concepts + Equation Index → Target):")
    print(f"  Train  - MSE: {train_loss_ce2t:.4f}, MAE: {train_mae_ce2t:.4f}, R²: {train_r2_ce2t:.4f}")
    print(f"  Val    - MSE: {val_loss_ce2t:.4f}, MAE: {val_mae_ce2t:.4f}, R²: {val_r2_ce2t:.4f}")
    print(f"  Test   - MSE: {test_loss_ce2t:.4f}, MAE: {test_mae_ce2t:.4f}, R²: {test_r2_ce2t:.4f}")
    print("="*80)
    
    # Display sample examples from each split
    print("\n" + "="*80)
    print("SAMPLE EXAMPLES (Ground Truth)")
    print("="*80)
    
    def display_samples(df, split_name, num_samples=10):
        """Display sample examples from a split."""
        print(f"\n--- {split_name} Split (10 samples) ---\n")
        
        # Take first 10 samples or all if less than 10
        samples = df.head(num_samples)
        
        for idx, (_, row) in enumerate(samples.iterrows(), 1):
            print(f"Sample {idx}:")
            print(f"  Question: {row['Question']}")
            print(f"  Equation: {row['Equation']}")
            print(f"  Concepts: N_00={row['N_00']:.2f}, N_01={row['N_01']:.2f}, N_02={row['N_02']:.2f}")
            print(f"  Target:   {row['Answer']:.2f}")
            print()
    
    display_samples(train_df, "Train")
    display_samples(val_df, "Validation")
    display_samples(test_df, "Test")
    
    print("="*80)


if __name__ == "__main__":
    main()
