"""
Test script to evaluate the quality of TimeSformer embeddings for synthetic motion dataset.
Trains an MLP to predict concepts and targets from embeddings.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import os

# -----------------------------
# MODEL DEFINITION
# -----------------------------
class EmbeddingPredictor(nn.Module):
    """MLP to predict concepts and target from video embeddings"""
    def __init__(self, embedding_dim=768, concept_dim=5, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = embedding_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Separate heads for concepts and target
        self.concept_head = nn.Linear(prev_dim, concept_dim)
        self.target_head = nn.Linear(prev_dim, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        concepts = self.concept_head(features)
        target = self.target_head(features).squeeze(-1)
        return concepts, target


# -----------------------------
# TRAINING FUNCTIONS
# -----------------------------
def train_epoch(model, loader, optimizer, criterion_concepts, criterion_target, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    concept_loss_sum = 0
    target_loss_sum = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        x = batch['x'].to(device)
        c = batch['c'].to(device)
        y = batch['y'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_c, pred_y = model(x)
        
        # Compute losses
        loss_c = criterion_concepts(pred_c, c)
        loss_y = criterion_target(pred_y, y)
        loss = loss_c + loss_y
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        concept_loss_sum += loss_c.item()
        target_loss_sum += loss_y.item()
    
    n_batches = len(loader)
    return total_loss / n_batches, concept_loss_sum / n_batches, target_loss_sum / n_batches


def evaluate(model, loader, criterion_concepts, criterion_target, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    concept_loss_sum = 0
    target_loss_sum = 0
    
    # For computing metrics
    all_pred_c = []
    all_true_c = []
    all_pred_y = []
    all_true_y = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            x = batch['x'].to(device)
            c = batch['c'].to(device)
            y = batch['y'].to(device)
            
            # Forward pass
            pred_c, pred_y = model(x)
            
            # Compute losses
            loss_c = criterion_concepts(pred_c, c)
            loss_y = criterion_target(pred_y, y)
            loss = loss_c + loss_y
            
            total_loss += loss.item()
            concept_loss_sum += loss_c.item()
            target_loss_sum += loss_y.item()
            
            # Store predictions
            all_pred_c.append(pred_c.cpu())
            all_true_c.append(c.cpu())
            all_pred_y.append(pred_y.cpu())
            all_true_y.append(y.cpu())
    
    # Concatenate all predictions
    all_pred_c = torch.cat(all_pred_c)
    all_true_c = torch.cat(all_true_c)
    all_pred_y = torch.cat(all_pred_y)
    all_true_y = torch.cat(all_true_y)
    
    # Compute metrics
    concept_mae = torch.abs(all_pred_c - all_true_c).mean(dim=0)
    target_mae = torch.abs(all_pred_y - all_true_y).mean()
    
    # R^2 score for target
    ss_res = ((all_true_y - all_pred_y) ** 2).sum()
    ss_tot = ((all_true_y - all_true_y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    
    n_batches = len(loader)
    metrics = {
        'total_loss': total_loss / n_batches,
        'concept_loss': concept_loss_sum / n_batches,
        'target_loss': target_loss_sum / n_batches,
        'concept_mae': concept_mae.numpy(),
        'target_mae': target_mae.item(),
        'target_r2': r2.item()
    }
    
    return metrics


# -----------------------------
# MAIN
# -----------------------------
def main(args):
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load cached data
    cache_dir = "/home/fdesantis/.cache/linear_memory_reasoner/stored_tensors/embeddings/synthetic_motion/sentence-transformers_all-MiniLM-L6-v2/seed_1"
    print(f"\nLoading cached data from: {cache_dir}")
    
    train_loader = torch.load(os.path.join(cache_dir, "train.pt"))
    val_loader = torch.load(os.path.join(cache_dir, "val.pt"))
    test_loader = torch.load(os.path.join(cache_dir, "test.pt"))
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get embedding dimension from first batch
    first_batch = next(iter(train_loader))
    embedding_dim = first_batch['x'].shape[1]
    concept_dim = first_batch['c'].shape[1]
    
    print(f"\nEmbedding dimension: {embedding_dim}")
    print(f"Concept dimension: {concept_dim}")
    print(f"Target dimension: 1 (scalar)")
    
    # Initialize model
    model = EmbeddingPredictor(
        embedding_dim=embedding_dim,
        concept_dim=concept_dim,
        hidden_dims=args.hidden_dims
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss functions
    criterion_concepts = nn.MSELoss()
    criterion_target = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True
    )
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*70}\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_c_loss, train_y_loss = train_epoch(
            model, train_loader, optimizer, criterion_concepts, criterion_target, device
        )
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion_concepts, criterion_target, device
        )
        
        # Update learning rate
        scheduler.step(val_metrics['total_loss'])
        
        # Print metrics
        print(f"  Train Loss: {train_loss:.6f} (C: {train_c_loss:.6f}, Y: {train_y_loss:.6f})")
        print(f"  Val Loss:   {val_metrics['total_loss']:.6f} (C: {val_metrics['concept_loss']:.6f}, Y: {val_metrics['target_loss']:.6f})")
        print(f"  Val MAE - Concepts: {val_metrics['concept_mae']}")
        print(f"  Val MAE - Target: {val_metrics['target_mae']:.6f}")
        print(f"  Val R² - Target: {val_metrics['target_r2']:.4f}")
        
        # Early stopping
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_embedding_predictor.pt')
            print(f"  ✓ New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print()
    
    # Load best model for final evaluation
    print(f"\n{'='*70}")
    print("Loading best model for final evaluation")
    print(f"{'='*70}\n")
    model.load_state_dict(torch.load('best_embedding_predictor.pt'))
    
    # Final evaluation on validation set
    print("Validation Set Results:")
    val_metrics = evaluate(model, val_loader, criterion_concepts, criterion_target, device)
    print(f"  Loss: {val_metrics['total_loss']:.6f}")
    print(f"  Concept MAE: {val_metrics['concept_mae']}")
    print(f"  Target MAE: {val_metrics['target_mae']:.6f}")
    print(f"  Target R²: {val_metrics['target_r2']:.4f}")
    
    # Final evaluation on test set
    print("\nTest Set Results:")
    test_metrics = evaluate(model, test_loader, criterion_concepts, criterion_target, device)
    print(f"  Loss: {test_metrics['total_loss']:.6f}")
    print(f"  Concept MAE: {test_metrics['concept_mae']}")
    print(f"  Target MAE: {test_metrics['target_mae']:.6f}")
    print(f"  Target R²: {test_metrics['target_r2']:.4f}")
    
    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"{'='*70}")
    
    # Interpretation
    print("\nInterpretation:")
    print(f"  - Lower MAE is better (measures average prediction error)")
    print(f"  - R² closer to 1.0 is better (measures explained variance)")
    print(f"  - Concept MAE < 1.0 is good (concepts are normalized to [0, 10])")
    print(f"  - If Target R² > 0.8, embeddings capture motion information well")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test embedding quality for synthetic motion dataset')
    
    # Data
    parser.add_argument('--cache_dir', type=str, 
                        default='/home/fdesantis/.cache/linear_memory_reasoner/stored_tensors/embeddings/synthetic_motion/seed_1',
                        help='Path to cached tensor directory')
    
    # Model
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256, 128],
                        help='Hidden layer dimensions')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Patience for early stopping')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index (-1 for CPU)')
    
    args = parser.parse_args()
    
    main(args)
