import os
import torch
import numpy as np
import random
from omegaconf import DictConfig, OmegaConf, open_dict
from time import time
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import transformers
from env import CACHE
import sympy as sp
import warnings
warnings.filterwarnings("ignore")
import re

# DO NOT import PySRRegressor at module level - it will be imported lazily when needed
# from pysr import PySRRegressor

# warnings.filterwarnings("ignore")

def set_matmul_precision():
    """
    Set float32 matmul precision for better performance on CUDA devices with Tensor Cores.
    Automatically detects the GPU and sets precision only if Tensor Cores are available.
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        # List of GPUs with Tensor Cores that benefit from reduced precision
        tensor_core_gpus = ['V100', 'A100', 'A6000', 'A5000', 'A4000', 'RTX', 'T4', 'H100']
        if any(gpu in device_name for gpu in tensor_core_gpus):
            torch.set_float32_matmul_precision('medium')
            print(f"Setting matmul precision to 'medium' for {device_name}")
        else:
            print(f"Using default matmul precision for {device_name}")

def set_seed(seed: int):
    print(f"Seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_valid_experiment(cfg: DictConfig):
    """ 
    Check if the experiment is valid based on the dataset and model combination.
    """
    if cfg.dataset.metadata.name in ['cebab', 'mnist_arithmetic', 'dsprites_simple', 'dsprites_hard'] and cfg.model.metadata.name in ['cem', 'licem', 'dcr', 'cmr']:
        raise ValueError(f"The experiment is not valid. Please check the configuration.\n\
                         The combination of {cfg.dataset.metadata.name}, {cfg.model.metadata.name} cannot be executed.")

def set_loggers(cfg):
    """ Set the loggers for the experiment """
    # Update the note in the config: if it is None, set it to an empty string
    with open_dict(cfg):
        cfg.update(
            note = "_" if cfg.note is None else "_"+str(cfg.note)
        )
    name = f"{cfg.dataset.metadata.name}.{cfg.model.metadata.name}.{cfg.seed}.{int(time())}"
    group_format = (
        "{dataset}_"
        "{model}"
        "{note}"
    )
    # Define the tags for wandb
    tags = [cfg.dataset.metadata.name, cfg.model.metadata.name, cfg.note]
    # Define the group for wandb
    group = group_format.format(**parse_hyperparams(cfg))
    if cfg.wandb.project is None or cfg.wandb.entity is None:
        wandb_logger = None
    else:
        wandb_logger = WandbLogger(project=cfg.wandb.project, 
                               entity=cfg.wandb.entity, 
                               name=name,
                               group=group,
                               tags=tags)
        wandb_logger.log_hyperparams(parse_hyperparams(cfg))
    csv_logger = CSVLogger("logs/",
                           name="experiment_metrics")
    return wandb_logger, csv_logger

def parse_hyperparams(cfg: DictConfig):
    hyperparams = {
        "dataset": cfg.dataset.metadata.name,
        "model": cfg.model.metadata.name,
        "seed": cfg.seed,
        "hydra_cfg": OmegaConf.to_container(cfg),
        "note": cfg.note,
    }
    return hyperparams

def get_backbone_latent_size(backbone):
    if backbone == 'resnet18':
        model = resnet18(pretrained=True)
    elif backbone == 'resnet34':
        model = resnet34(pretrained=True)
    elif backbone == 'resnet50':
        model = resnet50(pretrained=True)
    elif backbone == 'resnet101':
        model = resnet101(pretrained=True)
    elif backbone == 'resnet152':
        model = resnet152(pretrained=True)
    elif 'vit' in backbone or 'dino' in backbone:
        model = transformers.ViTModel.from_pretrained(backbone)
    elif backbone == 'bert-base-uncased':
        model = transformers.AutoModel.from_pretrained(backbone)
        return 768  # BERT base model has 768 hidden size
    elif backbone == 'sentence-transformers/all-mpnet-base-v2':
        return 768  # all-mpnet-base-v2 model has 768 hidden size
    elif backbone == 'sentence-transformers/all-MiniLM-L6-v2':
        return 384  # all-MiniLM-L6-v2 model has 384 hidden size
    else:
        raise ValueError(f"Image backbone {backbone} not recognized.")
    
    if 'resnet' in backbone:
        model = nn.Sequential(*list(model.children())[:-1])
        test = model(torch.randn((1,3,224,224)))
        latent_dim = test.flatten(start_dim=1).shape[1]
    elif 'vit' in backbone or 'dino' in backbone:
        test = model(torch.randn((1,3,224,224)))
        latent_dim = test.last_hidden_state[:, 0, :].shape[1]
    else:
        pass 
    
    # delete the model to free memory
    del model
    del test
    torch.cuda.empty_cache()
    return latent_dim

def get_type_from_name(dataset_name):
    # Symbolic regression datasets
    symbolic_prefixes = ['feynman_']
    if any(dataset_name.startswith(prefix) for prefix in symbolic_prefixes):
        return 'symbolic_regression'
    
    # Image datasets
    if dataset_name in ['mnist_addition', 'cub', 'cub_incomplete', \
                        'awa2', 'awa2_incomplete', 'xor', 'celeba', \
                        'cifar10', 'cifar100', 'mnist_arithmetic', 'mnist_arithmetic_hard', \
                        'pendulum', 'dsprites', 'dsprites_simple', 'dsprites_complex', \
                        'mnist_exponential']:
        return 'image'
    
    # Text datasets (default)
    return 'text'

def get_batch_from_loader(train_loader, device):
    with torch.cuda.device(device if device != 'cpu' else 'cpu'):
        # Temporarily set default tensor type to CPU to avoid automatic GPU allocation
        original_default_tensor_type = torch.get_default_dtype()
        if device == 'cpu':
            torch.set_default_tensor_type('torch.FloatTensor')
        
        batch = next(iter(train_loader))
        
        # Restore original tensor type
        torch.set_default_dtype(original_default_tensor_type)
    return batch

def setup_encoder(cfg: DictConfig, input_size: int, backbone_latent_size: int) -> DictConfig:

    type = None

    # if we want to extract the embeddings it means that we are NOT 
    # fine-tuning a pre-trained backbone during training.
    # This means that we just need a linear encoder.
    if cfg.extract_embeddings:
        input_size = input_size if cfg.dataset.metadata.name != 'xor' else 2 
        cfg.model.params.encoder = {
            '_target_': 'src.models.encoders.mlp.MLPEncoder',
            'output_size': backbone_latent_size, 
            'activation': cfg.activation,
            'input_transform': {
                '_target_': 'src.models.encoders.transform.FlattenTransform',
            },
            'dropout': 0.1
        }

    else:
        backbone = cfg.img_backbone_name if get_type_from_name(cfg.dataset.metadata.name) == 'image' \
                                            else cfg.text_backbone_name
        
        if cfg.dataset.metadata.data_type == 'toy':
            target = 'src.models.encoders.linear.LinearEncoder'
            transform = None
        else:
            if 'vit' in backbone:
                target = 'src.models.encoders.vit.VitEncoder'
                transform = {
                    '_target_': "src.models.encoders.transform.VitTransform",
                    'flatten': False
                }
            elif 'resnet' in backbone:
                target = 'src.models.encoders.resnet.ResNetEncoder'
                transform = {
                    '_target_': "src.models.encoders.transform.ImageTransform",
                    'flatten': False
                }
            elif ('sentence-transformers' in backbone) or ('bert' in backbone):
                target = 'src.models.encoders.transformer.TransformerEncoder'
                transform = None
                type = backbone

        # If the dataset is mnist_addition, then perform its own preprocessing
        if cfg.dataset.metadata.name == 'mnist_addition':
            transform = {
                '_target_': "src.models.encoders.transform.MNISTTransform",
                'flatten': False
            }

        # If we are fine-tuning a pre-trained model,
        # we need to set the encoder to the one defined in the dataset config.
        cfg.model.params.encoder = {
            '_target_': target,
            'output_size': backbone_latent_size, # we do not want the linear layer to reduce the size of the embeddings
            'input_transform' : transform,
        } 

    with open_dict(cfg):
        cfg.model.params.encoder.update(
            input_size = input_size,
        )

        if type != None:
            cfg.model.params.encoder.update(
                type = type
            )

    return cfg

def allow_hard_concepts(dataset_name):
    # Datasets that do not allow hard concepts so far: cebab, mnist_arithmetic.
    return dataset_name in ['xor', 'cub', 'mnist_addition', 'awa2', 'cub_incomplete', 'awa2_incomplete', 'cifar10', 'cifar100']


def update_config_from_data(cfg: DictConfig, train_loader, c_names,
                            y_names, c_groups, csv_log_dir) -> DictConfig:
    """
    Update the config with the input size, output size, and concept names.
    """
    # Create a temporary dataloader that loads data directly to the specified device
    # or force CPU loading to avoid automatic GPU allocation
    device = cfg.gpus[0]
    batch = get_batch_from_loader(train_loader, device)

    if get_type_from_name(cfg.dataset.metadata.name) == 'image':
        x = batch['x']
        data_type = 'image'
    else:
        x = batch['x'] if cfg.extract_embeddings else batch['x']['input_ids']
        data_type = 'text'

    input_size = torch.prod(torch.tensor(x.shape[1:])).item()
        
    # Clean up GPU memory
    del batch
    del x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n_labels = len(y_names)

    if c_groups is None or not isinstance(c_groups, dict):
        c_groups = c_groups
    else:
        c_groups = dict(c_groups)

    backbone = cfg.img_backbone_name if get_type_from_name(cfg.dataset.metadata.name) == 'image' \
                else cfg.dataset.metadata.pretrained_transformer if cfg.dataset.metadata.name == 'mawps' \
                                        else cfg.text_backbone_name
    backbone_latent_size = cfg.dataset.latent_size if cfg.extract_embeddings else get_backbone_latent_size(backbone)

    with open_dict(cfg):
        cfg.engine.update(
            c_names = c_names,
            y_name = y_names,
            csv_log_dir = csv_log_dir,
            data_type = data_type,
            dataset_name = cfg.dataset.metadata.name,
            scale_target = cfg.scale_target if 'scale_target' in cfg else True
        )

        hard_concepts = cfg.hard_concepts
        concept_type = prepare_concept_type(cfg.dataset.metadata.concept_type, cfg.engine.c_names, cfg.dataset.metadata.name)

        # If cfg.dataset.equations exists, then we want to use the known equations, 
        # null otherwise.
        known_equations = cfg.dataset.equations if 'equations' in cfg.dataset and cfg.dataset.equations is not None else None

        cfg.model.params.update(
            output_size = n_labels,
            c_names = c_names,
            y_names = y_names,
            task = cfg.dataset.metadata.task,
            c_groups = c_groups,
            backbone_latent_size = backbone_latent_size,
            concept_type = concept_type,
            hard_concepts = hard_concepts,
            known_equations = known_equations,
            disjoint_training = cfg.disjoint_training,
            device = device
        )

        cfg = setup_encoder(cfg, input_size, backbone_latent_size)

    return cfg

def prepare_concept_type(c_types, c_names, dataset_name):
    n_concepts = len(c_names)
    if isinstance(c_types, list):
        return c_types
    else:
        if dataset_name == 'mnist_exponential':
            return c_types
        return [c_types] * n_concepts

def generate_data_path(cfg):
    data_path = os.path.join(str(CACHE), 
                             'stored_tensors', 
                             'embeddings' if cfg.extract_embeddings else 'raw', # whether it contains embeddings or not
                             cfg.dataset.metadata.name)

    # Add backbone name
    dataset_type = get_type_from_name(cfg.dataset.metadata.name)
    
    if dataset_type == 'image':
        img_backbone_name = cfg.img_backbone_name.replace('/', '_')
        data_path += f"/{img_backbone_name}"
    elif dataset_type == 'text':
        text_backbone_name = cfg.text_backbone_name.replace('/', '_')
        data_path += f"/{text_backbone_name}"
    elif dataset_type == 'symbolic_regression':
        # For symbolic regression, use latent_dim and noise_std in the path
        latent_dim = cfg.dataset.loader.get('latent_dim', 4)
        noise_std = cfg.dataset.loader.get('noise_std', 0.0)
        data_path += f"/latent{latent_dim}_noise{str(noise_std).replace('.', '')}"

    # Add seed
    data_path += f"/seed_{cfg.seed}"

    if cfg.dataset.loader.concept_percentage != None:
        # Add concept percentage if it is not None
        data_path += f"_{str(cfg.dataset.loader.concept_percentage).replace('.', '')}"

    train_path = f"{data_path}/train.pt"
    val_path = f"{data_path}/val.pt"
    test_path = f"{data_path}/test.pt"

    return data_path, train_path, val_path, test_path

def generate_data_path(cfg):
    data_path = os.path.join(str(CACHE), 
                             'stored_tensors', 
                             'embeddings' if cfg.extract_embeddings else 'raw', # whether it contains embeddings or not
                             cfg.dataset.metadata.name)

    # Add backbone name
    img_backbone_name = cfg.img_backbone_name.replace('/', '_')
    text_backbone_name = cfg.text_backbone_name.replace('/', '_')

    data_path += f"/{img_backbone_name}" if get_type_from_name(cfg.dataset.metadata.name) == 'image' \
                                                    else f"/{text_backbone_name}"

    # Add seed
    data_path += f"/seed_{cfg.seed}"

    if cfg.dataset.loader.concept_percentage != None:
        # Add concept percentage if it is not None
        data_path += f"_{str(cfg.dataset.loader.concept_percentage).replace('.', '')}"

    train_path = f"{data_path}/train.pt"
    val_path = f"{data_path}/val.pt"
    test_path = f"{data_path}/test.pt"

    return data_path, train_path, val_path, test_path

def standardize_tensor(tensor, dim=0):
    """
    Standardize a tensor to have zero mean and unit variance.
    
    Args:
        tensor: Input tensor to standardize
    """
    mean = tensor.mean(dim=dim)
    std = tensor.std(dim=dim)

    # Add small epsilon to avoid division by zero
    eps = 1e-8
    standardized = (tensor - mean) / (std + eps)

    return standardized, mean, std


def sanitize_concept_names(concept_names):
    """
    Create SymPy-safe concept names for a list of concept names.

    This returns a tuple (safe_names, mapping) where safe_names is a list of
    names guaranteed to be safe as SymPy variable identifiers and mapping is
    a dict mapping original_name -> safe_name.

    Behavior:
      - Replace any character that is not alphanumeric or underscore with an
        underscore.
      - Collapse repeated underscores and strip leading/trailing underscores.
      - If the resulting name starts with a digit, prefix it with an underscore.
      - Ensure deterministic uniqueness by appending "_1", "_2", ... when
        collisions occur after sanitization.

    This preserves readable concept names (e.g. "eye-color" -> "eye_color",
    "age (yrs)" -> "age_yrs") rather than mapping to generic names like
    c0, c1.
    """
    safe_names = []
    mapping = {}
    seen = {}

    for name in concept_names:
        # Ensure string
        orig = name
        s = str(name)

        # Replace non-alphanumeric (_ allowed) with underscore
        s = re.sub(r'[^0-9A-Za-z_]+', '_', s)

        # Collapse multiple underscores
        s = re.sub(r'__+', '_', s)

        # Strip leading/trailing underscores
        s = s.strip('_')

        # If empty after stripping, use placeholder
        if s == '':
            s = 'x'

        # If starts with digit, prefix with underscore
        if re.match(r'^[0-9]', s):
            s = f"_{s}"

        # Ensure uniqueness deterministically
        base = s
        count = seen.get(base, 0)
        if count > 0:
            s = f"{base}_{count}"
        seen[base] = count + 1

        safe_names.append(s)
        mapping[orig] = s

    return safe_names, mapping

def save_licem_linear_coefficients(model, loaded_set, log_dir, split='train'):
    """
    iterate over the loaded training data to get the generated weights (model_out['weights'])
    """
    all_weights = []
    model.model.eval()

    if split=='train':
        epss = [0.0]  # only one value for training
    else:
        epss = np.linspace(0, 1, 10)
    for eps in epss:
        all_weights = []
        model.model.eps = eps
        with torch.no_grad():
            for batch in loaded_set:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                model_out = model.model(batch)
                all_weights.append(model_out['weights'].cpu())
        all_weights = torch.cat(all_weights, dim=0)  # concatenate all weights

        if split == 'train':
            torch.save(all_weights, f"{log_dir}/learned_linear_coefficients_train.pt")
        else:
            torch.save(all_weights, f"{log_dir}/learned_linear_coefficients_test_{str(round(eps,2)).replace('.', '')}.pt")



def symbolic_regression(
        stored_concepts, 
        stored_targets, 
        stored_selector_probs,
        memory_size,
        output_size,
        c_names,
        y_names,
        device,
        pysr_params,
    ):

    """
    Fine-tune the model by replacing each MLP in the BlackBoxPredictor with 
    symbolic equations discovered by PySR.
    
    This method:
    1. Collects stored concepts, targets, and selector probabilities
    2. For each memory slot and each output class, fits a PySR model
    3. Extracts the best equation
    4. Creates a SymbolicPredictor with all discovered equations
    """
    
    # Lazy import PySR only when this function is called
    # This avoids Julia initialization conflicts with PyTorch at import time
    import pysr
    
    # Save current CUDA visibility and temporarily hide GPUs from Julia to avoid conflicts
    original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPUs from Julia
    
    try:
        # Try to initialize Julia if not already done
        pysr.julia_helpers.init_julia(julia_project=None, quiet=False)
    except:
        pass  # Already initialized, that's fine
    
    from pysr import PySRRegressor
    
    # Restore original CUDA visibility
    if original_cuda_visible is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    
    if len(stored_concepts) == 0:
        raise ValueError("No stored data for fine-tuning. Run forward passes with store_for_finetuning=True first.")
    
    # Move all data to CPU and convert to numpy to avoid GPU conflicts with Julia
    stored_concepts = stored_concepts.cpu()
    stored_targets = stored_targets.cpu()
    stored_selector_probs = stored_selector_probs.cpu()

    # Handle different target shapes
    if stored_targets.ndim == 1:
        all_targets = all_targets.reshape(-1, 1)
    
    # Dictionary to store equations
    # Structure: {memory_idx: {output_name: sympy_equation}}
    all_equations = {i: {} for i in range(memory_size)}
    
    # For each memory slot
    for memory_idx in range(memory_size):
        # Get samples where this memory slot was selected
        # We take the most common selection across n_samples
        #memory_mask = (stored_selector_probs.mode(dim=1).values == memory_idx).numpy()
        memory_mask  = (stored_selector_probs.argmax(dim=1).flatten()==memory_idx).numpy()
        n_samples_for_memory = memory_mask.sum()
        
        # Filter concepts and targets for this memory slot
        X_memory = stored_concepts[memory_mask]  # [n_samples_memory, n_concepts]
        y_memory = stored_targets[memory_mask]    # [n_samples_memory, n_outputs]
        
        # Skip if no samples for this memory slot
        if n_samples_for_memory == 0:
            print(f"Memory slot {memory_idx} has no samples. Skipping.")
            for output_idx in range(output_size):
                output_name = y_names[output_idx] if output_idx < len(y_names) else f"y_{output_idx}"
                all_equations[memory_idx][output_name] = sp.sympify("0")
            continue
        
        # Note: you are running with more than 10,000 datapoints. 
        # You should consider turning on batching (`options.batching`), and also if you need that many datapoints. 
        # Unless you have a large amount of noise (in which case you should smooth your dataset first), 
        # generally < 10,000 datapoints is enough to find a functional form.
        # Given the message returned by PySR, we can subsample if needed.
        subsample_size = 5000
        if n_samples_for_memory > subsample_size:
            print(f"Subsampling to {subsample_size} for PySR.")
            indices = np.random.choice(n_samples_for_memory, size=subsample_size, replace=False)
            X_memory = X_memory[indices]
            y_memory = y_memory[indices]

        # For each output
        for output_idx in range(output_size):
            output_name = y_names[output_idx] if output_idx < len(y_names) else f"y_{output_idx}"
                            
            y_target = y_memory[:, output_idx]
            
            # Validate data: check for NaN and Inf values
            if np.isnan(X_memory).sum()>0 or np.isinf(X_memory).sum()>0:
                print(f"WARNING: X_memory contains NaN or Inf values for memory {memory_idx}. Cleaning data.")
                valid_mask = ~(np.isnan(X_memory).any(axis=1) | np.isinf(X_memory).any(axis=1))
                X_memory = X_memory[valid_mask]
                y_target = y_target[valid_mask]
            
            if np.isnan(y_target).sum()>0 or np.isinf(y_target).sum()>0:
                print(f"WARNING: y_target contains NaN or Inf values for memory {memory_idx}, output {output_name}. Cleaning data.")
                valid_mask = ~(np.isnan(y_target) | np.isinf(y_target))
                X_memory_clean = X_memory[valid_mask]
                y_target = y_target[valid_mask]
            else:
                X_memory_clean = X_memory
            
            print(f"\nFitting PySR for output '{output_name}' (memory slot {memory_idx})...")
            print(f"  Input shape: {X_memory_clean.shape}, Target shape: {y_target.shape}")
            
            try:

                model = PySRRegressor(
                    **pysr_params,
                    verbosity=1,  # Reduce verbosity to avoid Julia output issues
                    progress=True,
                )

                model.fit(X_memory_clean.cpu().numpy(), y_target.cpu().numpy())

                # Get the best equation (highest score)
                equations_df = model.equations_
                print(f"  Pareto front has {len(equations_df)} equations")
                
                # Select the best equation by score
                best_eq_row = equations_df.nlargest(1, 'score').iloc[0]
                sympy_eq = best_eq_row['sympy_format']
                
                # Rename variables from x0, x1, ... to concept names
                for i, c_name in enumerate(c_names):
                    sympy_eq = sympy_eq.subs(sp.Symbol(f'x{i}'), sp.Symbol(c_name))
                
                all_equations[memory_idx][output_name] = sympy_eq
                
                print(f"  ✓ Best equation: {sympy_eq}")
                print(f"    Loss: {best_eq_row['loss']:.6f}")
                print(f"    Complexity: {best_eq_row['complexity']}")
                print(f"    Score: {best_eq_row['score']:.6f}")

                # Clean up the model
                del model
                
            except Exception as e:
                print(f"  ✗ ERROR fitting PySR for memory {memory_idx}, output {output_name}:")
                print(f"    {type(e).__name__}: {str(e)}")
                print(f"    Using fallback constant equation (mean value)")
                if len(y_target) == 0:
                    print(f"    No samples available. Using zero as fallback.")
                    all_equations[memory_idx][output_name] = sp.sympify("0")
                else:
                    mean_val = float(y_target.mean())
                    all_equations[memory_idx][output_name] = sp.sympify(str(mean_val))
                    print(f"    Fallback equation: {mean_val}")

    # Verify all equations are present
    for memory_idx in range(memory_size):
        for output_idx in range(output_size):
            output_name = y_names[output_idx] if output_idx < len(y_names) else f"y_{output_idx}"
            if output_name not in all_equations[memory_idx]:
                raise ValueError(f"Missing equation for memory {memory_idx}, output {output_name}")
    
    return all_equations