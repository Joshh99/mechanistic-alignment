import os
import time
import json
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder"""
    # Architecture
    d_in: int = 768                    # Input dimension (transformer d_model)
    expansion_factor: int = 4          # Expansion factor (4x-8x d_model)
    
    # Training
    batch_size: int = 1024            # Batch size for training
    learning_rate: float = 1e-4       # Adam learning rate
    sparsity_coeff: float = 1e-3      # Initial sparsity coefficient �
    max_sparsity_coeff: float = 1e-1  # Maximum sparsity coefficient
    target_sparsity: float = 0.1      # Target fraction of active features (5-15%)
    
    # Training schedule
    n_epochs: int = 50                # Number of training epochs
    warmup_epochs: int = 10           # Epochs before sparsity annealing
    log_interval: int = 100           # Log every N steps
    
    # Dead neuron management
    dead_neuron_threshold: float = 0.01  # <1% activation frequency threshold
    dead_neuron_window: int = 1000       # Window for computing activation stats
    resample_dead_neurons: bool = True   # Enable dead neuron resampling
    
    # Device and precision
    device: str = "cpu"               # Device for computation
    dtype: torch.dtype = torch.float32  # Data type
    
    # Hook configuration
    hook_name: str = "blocks.2.hook_resid_post"  # Target hook for activation collection
    
    @property
    def d_sae(self) -> int:
        """SAE hidden dimension (expanded dimension)"""
        return self.d_in * self.expansion_factor
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        config_dict = asdict(self)
        # Handle non-serializable types
        config_dict['dtype'] = str(self.dtype)
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SAEConfig':
        """Load config from dictionary"""
        if 'dtype' in config_dict:
            if config_dict['dtype'] == 'torch.float32':
                config_dict['dtype'] = torch.float32
            elif config_dict['dtype'] == 'torch.float16':
                config_dict['dtype'] = torch.float16
        return cls(**config_dict)


class MiniSAE(nn.Module):
    """
    Mini Sparse Autoencoder for Mechanistic Interpretability
    
    Implements the core SAE architecture:
    h_reconstructed = W_dec * ReLU(W_enc * h + b_enc) + b_dec
    
    Features:
    - Configurable expansion factor (4x-8x)
    - Dead neuron detection and resampling
    - Feature activation statistics tracking
    - SAELens-compatible interface
    """
    
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        
        # Core architecture
        self.W_enc = nn.Parameter(torch.randn(cfg.d_sae, cfg.d_in))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        self.W_dec = nn.Parameter(torch.randn(cfg.d_sae, cfg.d_in))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))
        
        # Initialize weights
        self._init_weights()
        
        # Feature statistics for dead neuron detection
        self.register_buffer('feature_acts_history', torch.zeros(cfg.d_sae, cfg.dead_neuron_window))
        self.register_buffer('step_counter', torch.tensor(0))
        self.register_buffer('total_activations', torch.zeros(cfg.d_sae))
        self.register_buffer('activation_counts', torch.zeros(cfg.d_sae))
        
        # Training state
        self.training_step = 0
        self.dead_neurons_resampled = 0
        
    def _init_weights(self):
        """Initialize weights according to SAE best practices"""
        # Xavier initialization for encoder
        nn.init.xavier_uniform_(self.W_enc)
        
        # Unit norm rows for decoder (important for SAE stability)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)
        
        # Zero bias initialization
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_dec)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input activations to sparse features
        
        Args:
            x: Input activations [batch, d_in]
            
        Returns:
            Sparse feature activations [batch, d_sae]
        """
        return F.relu(F.linear(x, self.W_enc, self.b_enc))
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to original space
        
        Args:
            features: Sparse feature activations [batch, d_sae]
            
        Returns:
            Reconstructed activations [batch, d_in]
        """
        return F.linear(features, self.W_dec.T, self.b_dec)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode
        
        Args:
            x: Input activations [batch, d_in]
            
        Returns:
            (reconstructed_activations, sparse_features)
        """
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features
    
    def compute_loss(self, x: torch.Tensor, reconstruction: torch.Tensor, 
                    features: torch.Tensor, sparsity_coeff: float) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss: reconstruction + sparsity
        
        Args:
            x: Original activations [batch, d_in]
            reconstruction: Reconstructed activations [batch, d_in] 
            features: Sparse features [batch, d_sae]
            sparsity_coeff: lambda coefficient for sparsity penalty
            
        Returns:
            Dictionary of loss components
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstruction, x, reduction='mean')
        
        # Sparsity loss (L1 penalty on features)
        sparsity_loss = torch.mean(torch.abs(features))
        
        # Combined loss
        total_loss = reconstruction_loss + sparsity_coeff * sparsity_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'sparsity_loss': sparsity_loss,
            'sparsity_coeff': torch.tensor(sparsity_coeff, device=x.device)
        }
    
    def update_feature_stats(self, features: torch.Tensor):
        """
        Update feature activation statistics for dead neuron detection
        
        Args:
            features: Sparse feature activations [batch, d_sae]
        """
        with torch.no_grad():
            # Update activation counts
            active_features = (features > 0).float()  # Binary activation indicator
            batch_activation_freq = torch.mean(active_features, dim=0)  # [d_sae]
            
            # Update rolling history
            step_idx = self.step_counter % self.cfg.dead_neuron_window
            self.feature_acts_history[:, step_idx] = batch_activation_freq
            
            # Update cumulative stats
            self.total_activations += torch.sum(active_features, dim=0)
            self.activation_counts += features.shape[0]
            
            self.step_counter += 1
    
    def get_dead_neurons(self) -> torch.Tensor:
        """
        Identify dead neurons based on activation frequency
        
        Returns:
            Boolean tensor indicating dead neurons [d_sae]
        """
        if self.step_counter < self.cfg.dead_neuron_window:
            # Use cumulative stats if not enough history
            activation_freq = self.total_activations / self.activation_counts.clamp(min=1)
        else:
            # Use rolling window average
            activation_freq = torch.mean(self.feature_acts_history, dim=1)
        
        return activation_freq < self.cfg.dead_neuron_threshold
    
    def resample_dead_neurons(self, training_data: torch.Tensor, dead_mask: torch.Tensor):
        """
        Resample dead neurons based on reconstruction error
        
        Args:
            training_data: Recent training activations [n_samples, d_in]
            dead_mask: Boolean mask of dead neurons [d_sae]
        """
        n_dead = dead_mask.sum().item()
        if n_dead == 0:
            return
        
        with torch.no_grad():
            # Compute reconstruction errors for each sample
            reconstruction, features = self.forward(training_data)
            reconstruction_errors = torch.mean((training_data - reconstruction) ** 2, dim=1)
            
            # Sample new neuron directions from high-error examples
            high_error_indices = torch.topk(reconstruction_errors, min(n_dead * 2, len(reconstruction_errors))).indices
            high_error_samples = training_data[high_error_indices]
            
            # Reinitialize dead neurons
            dead_indices = torch.where(dead_mask)[0]
            for i, dead_idx in enumerate(dead_indices):
                if i < len(high_error_samples):
                    # Set encoder weights to high-error sample direction
                    sample_norm = torch.norm(high_error_samples[i])
                    if sample_norm > 0:
                        self.W_enc.data[dead_idx] = high_error_samples[i] / sample_norm
                        
                        # Set decoder to encoder weights for symmetry
                        self.W_dec.data[dead_idx, :] = self.W_enc.data[dead_idx, :]
                        
                        # Reset bias
                        self.b_enc.data[dead_idx] = 0.0
            
            # Reset statistics for resampled neurons
            self.feature_acts_history[dead_mask] = 0.0
            self.total_activations[dead_mask] = 0.0
            
            self.dead_neurons_resampled += n_dead
    
    def get_sparsity_metrics(self, features: torch.Tensor) -> Dict[str, float]:
        """
        Compute sparsity metrics for feature activations
        
        Args:
            features: Sparse feature activations [batch, d_sae]
            
        Returns:
            Dictionary of sparsity metrics
        """
        with torch.no_grad():
            # L0 norm (fraction of active features)
            active_features = (features > 0).float()
            l0_norm = torch.mean(torch.sum(active_features, dim=1))
            sparsity_l0 = l0_norm / features.shape[1]  # Fraction active
            
            # L1 norm 
            l1_norm = torch.mean(torch.sum(torch.abs(features), dim=1))
            
            # Max activation
            max_activation = torch.max(features).item()
            
            # Feature usage (fraction of features used across batch)
            feature_usage = torch.mean((torch.sum(active_features, dim=0) > 0).float())
            
            return {
                'sparsity_l0': sparsity_l0.item(),
                'l0_norm': l0_norm.item(),
                'l1_norm': l1_norm.item(),
                'max_activation': max_activation,
                'feature_usage': feature_usage.item()
            }
    
    def get_reconstruction_metrics(self, x: torch.Tensor, reconstruction: torch.Tensor) -> Dict[str, float]:
        """
        Compute reconstruction quality metrics
        
        Args:
            x: Original activations [batch, d_in]
            reconstruction: Reconstructed activations [batch, d_in]
            
        Returns:
            Dictionary of reconstruction metrics  
        """
        with torch.no_grad():
            # Mean squared error
            mse = F.mse_loss(reconstruction, x).item()
            
            # Mean absolute error
            mae = F.l1_loss(reconstruction, x).item()
            
            # Cosine similarity
            cos_sim = F.cosine_similarity(x, reconstruction, dim=1).mean().item()
            
            # Explained variance
            var_orig = torch.var(x)
            var_residual = torch.var(x - reconstruction)
            explained_variance = (1 - var_residual / var_orig.clamp(min=1e-8)).item()
            
            return {
                'mse': mse,
                'mae': mae, 
                'cosine_similarity': cos_sim,
                'explained_variance': explained_variance
            }
    
    def normalize_decoder_weights(self):
        """Normalize decoder weight rows to unit norm (important for SAE stability)"""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)
    
    def get_feature_norms(self) -> torch.Tensor:
        """Get L2 norms of decoder weight rows"""
        return torch.norm(self.W_dec.data, dim=1)
    
    def save_checkpoint(self, path: str, optimizer_state: Optional[dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'cfg': self.cfg.to_dict(),
            'model_state_dict': self.state_dict(),
            'training_step': self.training_step,
            'dead_neurons_resampled': self.dead_neurons_resampled
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cpu') -> Tuple['MiniSAE', dict]:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        
        cfg = SAEConfig.from_dict(checkpoint['cfg'])
        cfg.device = device
        
        model = cls(cfg)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.training_step = checkpoint.get('training_step', 0)
        model.dead_neurons_resampled = checkpoint.get('dead_neurons_resampled', 0)
        
        optimizer_state = checkpoint.get('optimizer_state_dict', None)
        
        return model, optimizer_state


class ActivationCollector:
    """
    Collects activations from transformer hooks for SAE training
    
    Handles:
    - Hook registration and management
    - Batch collection and processing
    - Shape handling [batch, seq_len, d_model] -> [batch*seq_len, d_model]
    - Device consistency
    """
    
    def __init__(self, model, hook_name: str, max_samples: int = 100000):
        """
        Initialize activation collector
        
        Args:
            model: HookedTransformerWrapper instance
            hook_name: Name of hook to collect from
            max_samples: Maximum number of activation vectors to collect
        """
        self.model = model
        self.hook_name = hook_name
        self.max_samples = max_samples
        self.activations = []
        self.collected_samples = 0
        
    def collection_hook(self, activation: torch.Tensor, hook_name: str) -> torch.Tensor:
        """Hook function to collect activations"""
        if self.collected_samples < self.max_samples:
            # Flatten [batch, seq_len, d_model] -> [batch*seq_len, d_model]
            flat_activation = activation.reshape(-1, activation.shape[-1])
            
            # Store on CPU to save GPU memory
            self.activations.append(flat_activation.detach().cpu())
            self.collected_samples += flat_activation.shape[0]
        
        return activation
    
    def collect_activations(self, tokens_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Collect activations from multiple batches of tokens
        
        Args:
            tokens_list: List of token tensors to process
            
        Returns:
            Collected activations [n_samples, d_model]
        """
        print(f"Collecting activations from hook: {self.hook_name}")
        
        # Register collection hook
        hook_points = self.model._get_hook_points()
        if self.hook_name not in hook_points:
            raise ValueError(f"Hook {self.hook_name} not found in model")
        
        hook_point = hook_points[self.hook_name]
        hook_point.add_hook(self.collection_hook)
        
        try:
            # Process batches
            for i, tokens in enumerate(tokens_list):
                if self.collected_samples >= self.max_samples:
                    break
                
                with torch.no_grad():
                    _ = self.model(tokens)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed batch {i+1}/{len(tokens_list)}, collected {self.collected_samples} samples")
            
            # Combine all collected activations
            if self.activations:
                all_activations = torch.cat(self.activations, dim=0)
                # Truncate to max_samples if needed
                if all_activations.shape[0] > self.max_samples:
                    all_activations = all_activations[:self.max_samples]
                
                print(f"Collection complete: {all_activations.shape[0]} activation vectors")
                return all_activations
            else:
                raise ValueError("No activations collected")
                
        finally:
            # Clean up hook
            hook_point.remove_hooks()
    
    def clear_cache(self):
        """Clear collected activations to free memory"""
        self.activations = []
        self.collected_samples = 0