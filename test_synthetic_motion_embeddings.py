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
import matplotlib.pyplot as plt

# -----------------------------
# EQUATION EXECUTION
# -----------------------------
def execute_equation(concepts, equation_type):
    """
    Execute the motion equation given concepts.
    
    Args:
        concepts: tensor of shape (batch_size, 5) containing [x0, v0, a, t, theta]
        equation_type: tensor of shape (batch_size,) with 0=uniform, 1=accelerated
    
    Returns:
        target: tensor of shape (batch_size,) with computed position
    """
    x0 = concepts[:, 0]  # initial position
    v0 = concepts[:, 1]  # initial velocity
    a = concepts[:, 2]   # acceleration
    t = concepts[:, 3]   # time
    # theta = concepts[:, 4]  # angle (not used in 1D motion)
    
    # Compute position using the appropriate equation
    # Uniform motion: x = x0 + v0*t
    # Accelerated motion: x = x0 + v0*t + 0.5*a*t^2
    
    uniform_result = x0 + v0 * t
    accelerated_result = x0 + v0 * t + 0.5 * a * t**2
    
    # Select result based on equation type
    # equation_type: 0 for uniform, 1 for accelerated
    if equation_type.dim() == 0:  # scalar
        result = accelerated_result if equation_type.item() == 1 else uniform_result
    else:  # batch
        result = torch.where(equation_type == 1, accelerated_result, uniform_result)
    
    return result


# -----------------------------
# MODEL DEFINITION
# -----------------------------
class EmbeddingPredictor(nn.Module):
    """MLP to predict concepts, target, and equation from video embeddings"""
    def __init__(self, embedding_dim=768, concept_dim=5, num_equations=2, hidden_dims=[512, 256, 128]):
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
        
        # Separate heads for concepts, target, and equation
        self.concept_head = nn.Linear(prev_dim, concept_dim)
        self.target_head = nn.Linear(prev_dim, 1)
        self.equation_head = nn.Linear(prev_dim, num_equations)
    
    def forward(self, x):
        features = self.backbone(x)
        concepts = self.concept_head(features)
        target = self.target_head(features).squeeze(-1)
        equation = self.equation_head(features)
        return concepts, target, equation
    
    def forward_with_equation_execution(self, x):
        """Forward pass that computes target using equation execution"""
        features = self.backbone(x)
        concepts = self.concept_head(features)
        equation_logits = self.equation_head(features)
        
        # Get predicted equation type (hard selection)
        equation_type = equation_logits.argmax(dim=1)
        
        # Execute equation with predicted concepts
        target = execute_equation(concepts, equation_type)
        
        return concepts, target, equation_logits


# -----------------------------
# TRAINING FUNCTIONS
# -----------------------------
def train_epoch(model, loader, optimizer, criterion_concepts, criterion_target, criterion_equation, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    concept_loss_sum = 0
    target_loss_sum = 0
    equation_loss_sum = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        x = batch['x'].to(device)
        c = batch['c'].to(device)
        y = batch['y'].to(device)
        eq = batch['equation'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_c, pred_y, pred_eq = model(x)
        
        # Compute losses
        loss_c = criterion_concepts(pred_c, c)
        loss_y = criterion_target(pred_y, y)
        loss_eq = criterion_equation(pred_eq, eq)
        loss = loss_c + loss_y + loss_eq
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        concept_loss_sum += loss_c.item()
        target_loss_sum += loss_y.item()
        equation_loss_sum += loss_eq.item()
    
    n_batches = len(loader)
    return total_loss / n_batches, concept_loss_sum / n_batches, target_loss_sum / n_batches, equation_loss_sum / n_batches


def evaluate(model, loader, criterion_concepts, criterion_target, criterion_equation, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    concept_loss_sum = 0
    target_loss_sum = 0
    equation_loss_sum = 0
    
    # For computing metrics
    all_pred_c = []
    all_true_c = []
    all_pred_y = []
    all_true_y = []
    all_pred_eq = []
    all_true_eq = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            x = batch['x'].to(device)
            c = batch['c'].to(device)
            y = batch['y'].to(device)
            eq = batch['equation'].to(device)
            
            # Forward pass
            pred_c, pred_y, pred_eq = model(x)
            
            # Compute losses
            loss_c = criterion_concepts(pred_c, c)
            loss_y = criterion_target(pred_y, y)
            loss_eq = criterion_equation(pred_eq, eq)
            loss = loss_c + loss_y + loss_eq
            
            total_loss += loss.item()
            concept_loss_sum += loss_c.item()
            target_loss_sum += loss_y.item()
            equation_loss_sum += loss_eq.item()
            
            # Store predictions
            all_pred_c.append(pred_c.cpu())
            all_true_c.append(c.cpu())
            all_pred_y.append(pred_y.cpu())
            all_true_y.append(y.cpu())
            all_pred_eq.append(pred_eq.cpu())
            all_true_eq.append(eq.cpu())
    
    # Concatenate all predictions
    all_pred_c = torch.cat(all_pred_c)
    all_true_c = torch.cat(all_true_c)
    all_pred_y = torch.cat(all_pred_y)
    all_true_y = torch.cat(all_true_y)
    all_pred_eq = torch.cat(all_pred_eq)
    all_true_eq = torch.cat(all_true_eq)
    
    # Compute metrics
    concept_mae = torch.abs(all_pred_c - all_true_c).mean(dim=0)
    target_mae = torch.abs(all_pred_y - all_true_y).mean()
    
    # R^2 score for target
    ss_res = ((all_true_y - all_pred_y) ** 2).sum()
    ss_tot = ((all_true_y - all_true_y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    
    # Equation accuracy
    eq_pred_labels = all_pred_eq.argmax(dim=1)
    eq_accuracy = (eq_pred_labels == all_true_eq).float().mean()
    
    n_batches = len(loader)
    metrics = {
        'total_loss': total_loss / n_batches,
        'concept_loss': concept_loss_sum / n_batches,
        'target_loss': target_loss_sum / n_batches,
        'equation_loss': equation_loss_sum / n_batches,
        'concept_mae': concept_mae.numpy(),
        'target_mae': target_mae.item(),
        'target_r2': r2.item(),
        'equation_accuracy': eq_accuracy.item()
    }
    
    return metrics


def evaluate_with_equation_execution(model, loader, criterion_concepts, criterion_target, criterion_equation, device):
    """Evaluate the model using equation execution for target prediction"""
    model.eval()
    total_loss = 0
    concept_loss_sum = 0
    target_loss_sum = 0
    equation_loss_sum = 0
    
    # For computing metrics
    all_pred_c = []
    all_true_c = []
    all_pred_y = []
    all_true_y = []
    all_pred_eq = []
    all_true_eq = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating with equation execution", leave=False):
            x = batch['x'].to(device)
            c = batch['c'].to(device)
            y = batch['y'].to(device)
            eq = batch['equation'].to(device)
            
            # Forward pass with equation execution
            pred_c, pred_y, pred_eq = model.forward_with_equation_execution(x)
            
            # Compute losses
            loss_c = criterion_concepts(pred_c, c)
            loss_y = criterion_target(pred_y, y)
            loss_eq = criterion_equation(pred_eq, eq)
            loss = loss_c + loss_y + loss_eq
            
            total_loss += loss.item()
            concept_loss_sum += loss_c.item()
            target_loss_sum += loss_y.item()
            equation_loss_sum += loss_eq.item()
            
            # Store predictions
            all_pred_c.append(pred_c.cpu())
            all_true_c.append(c.cpu())
            all_pred_y.append(pred_y.cpu())
            all_true_y.append(y.cpu())
            all_pred_eq.append(pred_eq.cpu())
            all_true_eq.append(eq.cpu())
    
    # Concatenate all predictions
    all_pred_c = torch.cat(all_pred_c)
    all_true_c = torch.cat(all_true_c)
    all_pred_y = torch.cat(all_pred_y)
    all_true_y = torch.cat(all_true_y)
    all_pred_eq = torch.cat(all_pred_eq)
    all_true_eq = torch.cat(all_true_eq)
    
    # Compute metrics
    concept_mae = torch.abs(all_pred_c - all_true_c).mean(dim=0)
    target_mae = torch.abs(all_pred_y - all_true_y).mean()
    
    # R^2 score for target
    ss_res = ((all_true_y - all_pred_y) ** 2).sum()
    ss_tot = ((all_true_y - all_true_y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    
    # Equation accuracy
    eq_pred_labels = all_pred_eq.argmax(dim=1)
    eq_accuracy = (eq_pred_labels == all_true_eq).float().mean()
    
    n_batches = len(loader)
    metrics = {
        'total_loss': total_loss / n_batches,
        'concept_loss': concept_loss_sum / n_batches,
        'target_loss': target_loss_sum / n_batches,
        'equation_loss': equation_loss_sum / n_batches,
        'concept_mae': concept_mae.numpy(),
        'target_mae': target_mae.item(),
        'target_r2': r2.item(),
        'equation_accuracy': eq_accuracy.item()
    }
    
    return metrics


def intervention_experiment(model, loader, device, intervention_probs=None):
    """
    Perform intervention experiment: replace predicted concepts with ground truth
    with varying probability and measure the effect on target MAE.
    
    Args:
        model: trained model
        loader: data loader
        device: torch device
        intervention_probs: list of intervention probabilities to test
    
    Returns:
        results: dict mapping intervention probability to MAE
    """
    if intervention_probs is None:
        intervention_probs = np.linspace(0, 1, 11)  # 0%, 10%, ..., 100%
    
    model.eval()
    results = {}
    
    print(f"\nRunning intervention experiment with {len(intervention_probs)} probability levels...")
    
    for p_intervene in tqdm(intervention_probs, desc="Intervention probability"):
        all_pred_y = []
        all_true_y = []
        
        with torch.no_grad():
            for batch in loader:
                x = batch['x'].to(device)
                c_true = batch['c'].to(device)
                y_true = batch['y'].to(device)
                eq_true = batch['equation'].to(device)
                
                batch_size = x.shape[0]
                
                # Get predictions
                features = model.backbone(x)
                c_pred = model.concept_head(features)
                eq_logits = model.equation_head(features)
                eq_pred = eq_logits.argmax(dim=1)
                
                # Intervention: randomly replace predicted concepts with ground truth
                # Generate intervention mask (which samples to intervene on)
                intervene_mask = torch.rand(batch_size, device=device) < p_intervene
                
                # Create intervened concepts: use true concepts where mask is True
                c_intervened = torch.where(
                    intervene_mask.unsqueeze(1),  # broadcast to (batch_size, 1)
                    c_true,
                    c_pred
                )
                
                # Execute equation with intervened concepts
                y_pred = execute_equation(c_intervened, eq_pred)
                
                all_pred_y.append(y_pred.cpu())
                all_true_y.append(y_true.cpu())
        
        # Compute MAE for this intervention probability
        all_pred_y = torch.cat(all_pred_y)
        all_true_y = torch.cat(all_true_y)
        mae = torch.abs(all_pred_y - all_true_y).mean().item()
        
        results[p_intervene] = mae
        print(f"  p_intervene={p_intervene:.2f}: MAE={mae:.6f}")
    
    return results


def intervention_experiment_with_equation(model, loader, device, intervention_probs=None):
    """
    Perform intervention experiment: replace predicted concepts AND equations with ground truth
    with varying probability and measure the effect on target MAE.
    
    Args:
        model: trained model
        loader: data loader
        device: torch device
        intervention_probs: list of intervention probabilities to test
    
    Returns:
        results: dict mapping intervention probability to MAE
    """
    if intervention_probs is None:
        intervention_probs = np.linspace(0, 1, 11)  # 0%, 10%, ..., 100%
    
    model.eval()
    results = {}
    
    print(f"\nRunning intervention experiment (concepts + equation) with {len(intervention_probs)} probability levels...")
    
    for p_intervene in tqdm(intervention_probs, desc="Intervention probability"):
        all_pred_y = []
        all_true_y = []
        
        with torch.no_grad():
            for batch in loader:
                x = batch['x'].to(device)
                c_true = batch['c'].to(device)
                y_true = batch['y'].to(device)
                eq_true = batch['equation'].to(device)
                
                batch_size = x.shape[0]
                
                # Get predictions
                features = model.backbone(x)
                c_pred = model.concept_head(features)
                eq_logits = model.equation_head(features)
                eq_pred = eq_logits.argmax(dim=1)
                
                # Intervention: randomly replace predicted concepts AND equation with ground truth
                # Generate intervention mask (which samples to intervene on)
                intervene_mask = torch.rand(batch_size, device=device) < p_intervene
                
                # Create intervened concepts: use true concepts where mask is True
                c_intervened = torch.where(
                    intervene_mask.unsqueeze(1),  # broadcast to (batch_size, 1)
                    c_true,
                    c_pred
                )
                
                # Create intervened equation: use true equation where mask is True
                eq_intervened = torch.where(
                    intervene_mask,
                    eq_true,
                    eq_pred
                )
                
                # Execute equation with intervened concepts and equation
                y_pred = execute_equation(c_intervened, eq_intervened)
                
                all_pred_y.append(y_pred.cpu())
                all_true_y.append(y_true.cpu())
        
        # Compute MAE for this intervention probability
        all_pred_y = torch.cat(all_pred_y)
        all_true_y = torch.cat(all_true_y)
        mae = torch.abs(all_pred_y - all_true_y).mean().item()
        
        results[p_intervene] = mae
        print(f"  p_intervene={p_intervene:.2f}: MAE={mae:.6f}")
    
    return results


def plot_intervention_results(results, output_path='intervention_results.png'):
    """Plot intervention probability vs MAE"""
    probs = sorted(results.keys())
    maes = [results[p] for p in probs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(probs, maes, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Intervention Probability', fontsize=14)
    plt.ylabel('Target MAE', fontsize=14)
    plt.title('Effect of Concept Intervention on Target Prediction', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def plot_combined_intervention_results(results_concepts, results_concepts_equation, output_path='intervention_comparison.png'):
    """Plot both intervention experiments on the same plot"""
    probs_c = sorted(results_concepts.keys())
    maes_c = [results_concepts[p] for p in probs_c]
    
    probs_ce = sorted(results_concepts_equation.keys())
    maes_ce = [results_concepts_equation[p] for p in probs_ce]
    
    plt.figure(figsize=(12, 7))
    plt.plot(probs_c, maes_c, 'o-', linewidth=2, markersize=8, label='Intervene on Concepts Only')
    plt.plot(probs_ce, maes_ce, 's-', linewidth=2, markersize=8, label='Intervene on Concepts + Equation')
    plt.xlabel('Intervention Probability', fontsize=14)
    plt.ylabel('Target MAE', fontsize=14)
    plt.title('Effect of Interventions on Target Prediction', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_path}")
    plt.close()


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
    cache_dir = "/home/fdesantis/.cache/linear_memory_reasoner/stored_tensors/embeddings/synthetic_motion/facebook_dinov2-base/seed_1"
    print(f"\nLoading cached data from: {cache_dir}")
    
    train_loader = torch.load(os.path.join(cache_dir, "train.pt"))
    val_loader = torch.load(os.path.join(cache_dir, "val.pt"))
    test_loader = torch.load(os.path.join(cache_dir, "test.pt"))
    
    # Add equation labels to data loaders
    print("\nAdding equation labels from annotations...")
    from env import DATA_PATH
    import json
    dataset_dir = Path(DATA_PATH) / "synthetic_motion"
    
    def add_equation_labels(loader, dataset_dir):
        """Add equation labels (0=uniform, 1=accelerated) to batches"""
        new_batches = []
        for batch in loader:
            video_indices = batch['video_idx']
            equations = []
            for idx in video_indices:
                ann_path = dataset_dir / "annotations" / f"sample_{idx.item()}.json"
                with open(ann_path, 'r') as f:
                    ann = json.load(f)
                # 0 for uniform, 1 for accelerated
                eq_label = 0 if ann['motion_type'] == 'uniform' else 1
                equations.append(eq_label)
            batch['equation'] = torch.tensor(equations, dtype=torch.long)
            new_batches.append(batch)
        return new_batches
    
    train_loader = add_equation_labels(train_loader, dataset_dir)
    val_loader = add_equation_labels(val_loader, dataset_dir)
    test_loader = add_equation_labels(test_loader, dataset_dir)
    
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
    num_equations = 2  # 0=uniform, 1=accelerated
    model = EmbeddingPredictor(
        embedding_dim=embedding_dim,
        concept_dim=concept_dim,
        num_equations=num_equations,
        hidden_dims=args.hidden_dims
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss functions
    criterion_concepts = nn.MSELoss()
    criterion_target = nn.MSELoss()
    criterion_equation = nn.CrossEntropyLoss()
    
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
        train_loss, train_c_loss, train_y_loss, train_eq_loss = train_epoch(
            model, train_loader, optimizer, criterion_concepts, criterion_target, criterion_equation, device
        )
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion_concepts, criterion_target, criterion_equation, device
        )
        
        # Update learning rate
        scheduler.step(val_metrics['total_loss'])
        
        # Print metrics
        print(f"  Train Loss: {train_loss:.6f} (C: {train_c_loss:.6f}, Y: {train_y_loss:.6f}, Eq: {train_eq_loss:.6f})")
        print(f"  Val Loss:   {val_metrics['total_loss']:.6f} (C: {val_metrics['concept_loss']:.6f}, Y: {val_metrics['target_loss']:.6f}, Eq: {val_metrics['equation_loss']:.6f})")
        print(f"  Val MAE - Concepts: {val_metrics['concept_mae']}")
        print(f"  Val MAE - Target: {val_metrics['target_mae']:.6f}")
        print(f"  Val R² - Target: {val_metrics['target_r2']:.4f}")
        print(f"  Val Equation Accuracy: {val_metrics['equation_accuracy']:.4f}")
        
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
    val_metrics = evaluate(model, val_loader, criterion_concepts, criterion_target, criterion_equation, device)
    print(f"  Loss: {val_metrics['total_loss']:.6f}")
    print(f"  Concept MAE: {val_metrics['concept_mae']}")
    print(f"  Target MAE: {val_metrics['target_mae']:.6f}")
    print(f"  Target R²: {val_metrics['target_r2']:.4f}")
    print(f"  Equation Accuracy: {val_metrics['equation_accuracy']:.4f}")
    
    # Final evaluation on test set
    print("\nTest Set Results:")
    test_metrics = evaluate(model, test_loader, criterion_concepts, criterion_target, criterion_equation, device)
    print(f"  Loss: {test_metrics['total_loss']:.6f}")
    print(f"  Concept MAE: {test_metrics['concept_mae']}")
    print(f"  Target MAE: {test_metrics['target_mae']:.6f}")
    print(f"  Target R²: {test_metrics['target_r2']:.4f}")
    print(f"  Equation Accuracy: {test_metrics['equation_accuracy']:.4f}")
    
    # Evaluate with equation execution
    print(f"\n{'='*70}")
    print("Evaluation with Equation Execution")
    print(f"{'='*70}")
    
    print("\nValidation Set Results (with equation execution):")
    val_eq_metrics = evaluate_with_equation_execution(model, val_loader, criterion_concepts, criterion_target, criterion_equation, device)
    print(f"  Loss: {val_eq_metrics['total_loss']:.6f}")
    print(f"  Concept MAE: {val_eq_metrics['concept_mae']}")
    print(f"  Target MAE: {val_eq_metrics['target_mae']:.6f}")
    print(f"  Target R²: {val_eq_metrics['target_r2']:.4f}")
    print(f"  Equation Accuracy: {val_eq_metrics['equation_accuracy']:.4f}")
    
    print("\nTest Set Results (with equation execution):")
    test_eq_metrics = evaluate_with_equation_execution(model, test_loader, criterion_concepts, criterion_target, criterion_equation, device)
    print(f"  Loss: {test_eq_metrics['total_loss']:.6f}")
    print(f"  Concept MAE: {test_eq_metrics['concept_mae']}")
    print(f"  Target MAE: {test_eq_metrics['target_mae']:.6f}")
    print(f"  Target R²: {test_eq_metrics['target_r2']:.4f}")
    print(f"  Equation Accuracy: {test_eq_metrics['equation_accuracy']:.4f}")
    
    # Intervention experiments
    print(f"\n{'='*70}")
    print("Concept Intervention Experiments")
    print(f"{'='*70}")
    
    # Run intervention experiment on test set - concepts only
    intervention_probs = np.linspace(0, 1, 21)  # 0%, 5%, 10%, ..., 100%
    print("\n1. Intervening on CONCEPTS ONLY:")
    intervention_results_concepts = intervention_experiment(model, test_loader, device, intervention_probs)
    
    # Run intervention experiment on test set - concepts + equation
    print("\n2. Intervening on CONCEPTS + EQUATION:")
    intervention_results_concepts_equation = intervention_experiment_with_equation(model, test_loader, device, intervention_probs)
    
    # Plot results
    plot_intervention_results(intervention_results_concepts, output_path='intervention_concepts_only.png')
    plot_intervention_results(intervention_results_concepts_equation, output_path='intervention_concepts_equation.png')
    plot_combined_intervention_results(intervention_results_concepts, intervention_results_concepts_equation, 
                                       output_path='intervention_comparison.png')
    
    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"{'='*70}")
    
    # Interpretation
    print("\nInterpretation:")
    print(f"  - Lower MAE is better (measures average prediction error)")
    print(f"  - R² closer to 1.0 is better (measures explained variance)")
    print(f"  - Concept MAE < 1.0 is good (concepts are normalized to [0, 10])")
    print(f"  - If Target R² > 0.8, embeddings capture motion information well")
    print(f"  - Equation Accuracy > 0.9 means the model can distinguish motion types")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test embedding quality for synthetic motion dataset')
    
    # Data
    parser.add_argument('--cache_dir', type=str, 
                        default='/home/fdesantis/.cache/linear_memory_reasoner/stored_tensors/embeddings/synthetic_motion/facebook_dinov2-base/seed_1',
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
