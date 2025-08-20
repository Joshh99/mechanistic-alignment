import os
import time
import json
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .model import MiniSAE, SAEConfig, ActivationCollector


class SAETrainer:
    """
    SAE Training Manager
    
    Handles:
    - Efficient batch processing (1024-4096 activation vectors)
    - Training loop with loss tracking and logging
    - Sparsity coefficient annealing
    - Dead neuron detection and resampling
    - Model checkpointing and analysis
    """
    
    def __init__(self, sae: MiniSAE, cfg: SAEConfig):
        self.sae = sae
        self.cfg = cfg
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.sae.parameters(),
            lr=cfg.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Loss tracking
        self.loss_history = defaultdict(list)
        self.metrics_history = defaultdict(list)
        
        # Sparsity annealing
        self.current_sparsity_coeff = cfg.sparsity_coeff
        
        # Dead neuron tracking
        self.dead_neuron_history = []
        self.resample_history = []
        
    def compute_sparsity_coefficient(self, epoch: int, step: int) -> float:
        """
        Compute current sparsity coefficient with annealing
        
        Args:
            epoch: Current epoch
            step: Current step within epoch
            
        Returns:
            Current sparsity coefficient
        """
        if epoch < self.cfg.warmup_epochs:
            # Linear warmup
            progress = epoch / self.cfg.warmup_epochs
            return self.cfg.sparsity_coeff * progress
        else:
            # Exponential annealing towards target sparsity
            annealing_epochs = self.cfg.n_epochs - self.cfg.warmup_epochs
            progress = min(1.0, (epoch - self.cfg.warmup_epochs) / annealing_epochs)
            
            # Increase sparsity coefficient to reach target sparsity
            return self.cfg.sparsity_coeff + (self.cfg.max_sparsity_coeff - self.cfg.sparsity_coeff) * progress
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train SAE for one epoch
        
        Args:
            dataloader: DataLoader with activation vectors
            epoch: Current epoch number
            
        Returns:
            Epoch metrics
        """
        self.sae.train()
        epoch_losses = defaultdict(list)
        epoch_metrics = defaultdict(list)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, (batch_activations,) in enumerate(progress_bar):
            batch_activations = batch_activations.to(self.cfg.device, dtype=self.cfg.dtype)
            
            # Update sparsity coefficient
            self.current_sparsity_coeff = self.compute_sparsity_coefficient(epoch, step)
            
            # Forward pass
            reconstruction, features = self.sae(batch_activations)
            
            # Compute loss
            loss_dict = self.sae.compute_loss(
                batch_activations, 
                reconstruction, 
                features, 
                self.current_sparsity_coeff
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.sae.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Normalize decoder weights after each step
            self.sae.normalize_decoder_weights()
            
            # Update feature statistics
            self.sae.update_feature_stats(features.detach())
            
            # Compute metrics
            sparsity_metrics = self.sae.get_sparsity_metrics(features.detach())
            reconstruction_metrics = self.sae.get_reconstruction_metrics(
                batch_activations, reconstruction.detach()
            )
            
            # Log losses and metrics
            for key, value in loss_dict.items():
                epoch_losses[key].append(value.item())
            
            for key, value in sparsity_metrics.items():
                epoch_metrics[key].append(value)
                
            for key, value in reconstruction_metrics.items():
                epoch_metrics[key].append(value)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'recon': f"{loss_dict['reconstruction_loss'].item():.4f}",
                'sparsity': f"{sparsity_metrics['sparsity_l0']:.3f}",
                'lambda': f"{self.current_sparsity_coeff:.2e}"
            })
            
            # Dead neuron management
            if step % 500 == 0 and self.cfg.resample_dead_neurons:
                self.check_and_resample_dead_neurons(batch_activations)
            
            # Logging
            if step % self.cfg.log_interval == 0:
                self.log_training_step(epoch, step, loss_dict, sparsity_metrics, reconstruction_metrics)
            
            self.global_step += 1
        
        # Compute epoch averages
        epoch_summary = {}
        for key, values in epoch_losses.items():
            epoch_summary[f"loss_{key}"] = np.mean(values)
            
        for key, values in epoch_metrics.items():
            epoch_summary[f"metric_{key}"] = np.mean(values)
        
        return epoch_summary
    
    def check_and_resample_dead_neurons(self, recent_activations: torch.Tensor):
        """Check for dead neurons and resample if necessary"""
        dead_mask = self.sae.get_dead_neurons()
        n_dead = dead_mask.sum().item()
        
        if n_dead > 0:
            dead_fraction = n_dead / self.cfg.d_sae
            
            # Only resample if significant fraction are dead
            if dead_fraction > 0.2:  # >20% dead neurons
                print(f"\nResampling {n_dead} dead neurons ({dead_fraction:.1%} of features)")
                
                self.sae.resample_dead_neurons(recent_activations, dead_mask)
                
                self.resample_history.append({
                    'step': self.global_step,
                    'n_dead': n_dead,
                    'dead_fraction': dead_fraction
                })
        
        self.dead_neuron_history.append({
            'step': self.global_step,
            'n_dead': n_dead,
            'dead_fraction': dead_fraction
        })
    
    def log_training_step(self, epoch: int, step: int, loss_dict: Dict, 
                         sparsity_metrics: Dict, reconstruction_metrics: Dict):
        """Log training step information"""
        # Store in history
        for key, value in loss_dict.items():
            self.loss_history[key].append(value.item())
            
        for key, value in {**sparsity_metrics, **reconstruction_metrics}.items():
            self.metrics_history[key].append(value)
        
        self.metrics_history['sparsity_coeff'].append(self.current_sparsity_coeff)
        self.metrics_history['step'].append(self.global_step)
    
    def train(self, activations: torch.Tensor, validation_split: float = 0.1, 
              save_dir: Optional[str] = None) -> Dict[str, List]:
        """
        Full training loop
        
        Args:
            activations: Training activation vectors [n_samples, d_in]
            validation_split: Fraction of data to use for validation
            save_dir: Directory to save checkpoints and logs
            
        Returns:
            Training history
        """
        print(f"Starting SAE training on {activations.shape[0]} activation vectors")
        print(f"Architecture: {self.cfg.d_in} -> {self.cfg.d_sae} -> {self.cfg.d_in}")
        
        # Create save directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Split data
        n_train = int(activations.shape[0] * (1 - validation_split))
        train_data = activations[:n_train]
        val_data = activations[n_train:] if validation_split > 0 else None
        
        # Create dataloaders
        train_dataset = TensorDataset(train_data)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        if val_data is not None:
            val_dataset = TensorDataset(val_data)
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                drop_last=False
            )
        
        print(f"Training batches: {len(train_dataloader)}")
        if val_data is not None:
            print(f"Validation batches: {len(val_dataloader)}")
        
        # Training loop
        for epoch in range(self.cfg.n_epochs):
            start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader, epoch)
            
            # Validation
            if val_data is not None:
                val_metrics = self.validate(val_dataloader)
                print(f"\nValidation - Loss: {val_metrics['loss_total_loss']:.4f}, "
                      f"Sparsity: {val_metrics['metric_sparsity_l0']:.3f}")
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{self.cfg.n_epochs} ({epoch_time:.1f}s)")
            print(f"Train Loss: {train_metrics['loss_total_loss']:.4f} "
                  f"(Recon: {train_metrics['loss_reconstruction_loss']:.4f}, "
                  f"Sparsity: {train_metrics['loss_sparsity_loss']:.4f})")
            print(f"Sparsity: {train_metrics['metric_sparsity_l0']:.3f} "
                  f"(target: {self.cfg.target_sparsity:.3f}), "
                  f"lambda: {self.current_sparsity_coeff:.2e}")
            
            # Save checkpoint
            if save_dir and (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_dir, f"sae_checkpoint_epoch_{epoch+1}.pt")
                self.save_checkpoint(checkpoint_path)
            
            # Early stopping check
            current_loss = train_metrics['loss_total_loss']
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                if save_dir:
                    best_path = os.path.join(save_dir, "sae_best.pt")
                    self.save_checkpoint(best_path)
            
            self.current_epoch = epoch + 1
        
        # Final checkpoint
        if save_dir:
            final_path = os.path.join(save_dir, "sae_final.pt")
            self.save_checkpoint(final_path)
            
            # Save training history
            history_path = os.path.join(save_dir, "training_history.json")
            self.save_training_history(history_path)
        
        print(f"\nTraining complete! Dead neurons resampled: {self.sae.dead_neurons_resampled}")
        
        return {
            'loss_history': dict(self.loss_history),
            'metrics_history': dict(self.metrics_history),
            'dead_neuron_history': self.dead_neuron_history,
            'resample_history': self.resample_history
        }
    
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """Validate SAE on validation set"""
        self.sae.eval()
        val_losses = defaultdict(list)
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch_activations, in val_dataloader:
                batch_activations = batch_activations.to(self.cfg.device, dtype=self.cfg.dtype)
                
                # Forward pass
                reconstruction, features = self.sae(batch_activations)
                
                # Compute loss
                loss_dict = self.sae.compute_loss(
                    batch_activations, 
                    reconstruction, 
                    features, 
                    self.current_sparsity_coeff
                )
                
                # Compute metrics
                sparsity_metrics = self.sae.get_sparsity_metrics(features)
                reconstruction_metrics = self.sae.get_reconstruction_metrics(
                    batch_activations, reconstruction
                )
                
                # Accumulate
                for key, value in loss_dict.items():
                    val_losses[key].append(value.item())
                
                for key, value in sparsity_metrics.items():
                    val_metrics[key].append(value)
                    
                for key, value in reconstruction_metrics.items():
                    val_metrics[key].append(value)
        
        # Compute averages
        val_summary = {}
        for key, values in val_losses.items():
            val_summary[f"loss_{key}"] = np.mean(values)
            
        for key, values in val_metrics.items():
            val_summary[f"metric_{key}"] = np.mean(values)
        
        return val_summary
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        self.sae.save_checkpoint(path, self.optimizer.state_dict())
    
    def save_training_history(self, path: str):
        """Save training history to JSON"""
        history = {
            'loss_history': {k: [float(x) for x in v] for k, v in self.loss_history.items()},
            'metrics_history': {k: [float(x) for x in v] for k, v in self.metrics_history.items()},
            'dead_neuron_history': self.dead_neuron_history,
            'resample_history': self.resample_history,
            'config': self.cfg.to_dict()
        }
        
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)


class SAEAnalyzer:
    """
    Analysis tools for trained SAEs
    
    Features:
    - Max activating examples extraction
    - Feature correlation matrix computation  
    - Feature visualization and interpretation
    - Dashboard generation
    """
    
    def __init__(self, sae: MiniSAE, activations: torch.Tensor):
        """
        Initialize analyzer
        
        Args:
            sae: Trained SAE model
            activations: Dataset activations for analysis [n_samples, d_in]
        """
        self.sae = sae
        self.activations = activations.to(sae.cfg.device, dtype=sae.cfg.dtype)
        
        # Compute features for all activations
        with torch.no_grad():
            self.sae.eval()
            _, self.features = self.sae(self.activations)
    
    def get_max_activating_examples(self, feature_idx: int, n_examples: int = 10) -> Dict[str, Any]:
        """
        Get top examples that maximally activate a specific feature
        
        Args:
            feature_idx: Index of feature to analyze
            n_examples: Number of top examples to return
            
        Returns:
            Dictionary with top activations and indices
        """
        feature_activations = self.features[:, feature_idx]
        
        # Get top activating indices
        top_indices = torch.topk(feature_activations, n_examples).indices
        top_activations = feature_activations[top_indices]
        top_examples = self.activations[top_indices]
        
        return {
            'feature_idx': feature_idx,
            'top_indices': top_indices.cpu().numpy(),
            'top_activations': top_activations.cpu().numpy(),
            'top_examples': top_examples.cpu().numpy(),
            'mean_activation': feature_activations.mean().item(),
            'max_activation': feature_activations.max().item(),
            'activation_freq': (feature_activations > 0).float().mean().item()
        }
    
    def compute_feature_correlations(self, sample_size: Optional[int] = 10000) -> torch.Tensor:
        """
        Compute correlation matrix between features
        
        Args:
            sample_size: Number of samples to use (None for all)
            
        Returns:
            Feature correlation matrix [d_sae, d_sae]
        """
        features = self.features
        if sample_size and features.shape[0] > sample_size:
            # Sample random subset for efficiency
            indices = torch.randperm(features.shape[0])[:sample_size]
            features = features[indices]
        
        # Compute correlation matrix
        features_centered = features - features.mean(dim=0, keepdim=True)
        correlation_matrix = torch.corrcoef(features_centered.T)
        
        return correlation_matrix
    
    def analyze_feature_statistics(self) -> Dict[str, Any]:
        """
        Comprehensive feature statistics analysis
        
        Returns:
            Dictionary with various feature statistics
        """
        with torch.no_grad():
            # Basic activation statistics
            activation_freq = (self.features > 0).float().mean(dim=0)  # [d_sae]
            mean_activations = self.features.mean(dim=0)  # [d_sae]
            max_activations = self.features.max(dim=0).values  # [d_sae]
            
            # Dead/alive features
            dead_threshold = self.sae.cfg.dead_neuron_threshold
            alive_features = activation_freq > dead_threshold
            n_alive = alive_features.sum().item()
            
            # Sparsity distribution
            l0_norms = (self.features > 0).float().sum(dim=1)  # Per-sample sparsity
            
            # Feature importance (based on decoder norms)
            decoder_norms = torch.norm(self.sae.W_dec.data, dim=0)
            
            return {
                'n_features': self.sae.cfg.d_sae,
                'n_alive_features': n_alive,
                'fraction_alive': n_alive / self.sae.cfg.d_sae,
                'activation_freq_stats': {
                    'mean': activation_freq.mean().item(),
                    'std': activation_freq.std().item(), 
                    'min': activation_freq.min().item(),
                    'max': activation_freq.max().item()
                },
                'sparsity_stats': {
                    'mean_l0': l0_norms.mean().item(),
                    'std_l0': l0_norms.std().item(),
                    'target_sparsity': self.sae.cfg.target_sparsity * self.sae.cfg.d_sae
                },
                'decoder_norm_stats': {
                    'mean': decoder_norms.mean().item(),
                    'std': decoder_norms.std().item(),
                    'min': decoder_norms.min().item(),
                    'max': decoder_norms.max().item()
                }
            }
    
    def plot_training_curves(self, history: Dict, save_path: Optional[str] = None):
        """
        Plot training curves from training history
        
        Args:
            history: Training history from SAETrainer
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SAE Training Analysis', fontsize=16)
        
        steps = history['metrics_history']['step']
        
        # Loss curves
        axes[0, 0].plot(steps, history['loss_history']['total_loss'], label='Total', alpha=0.7)
        axes[0, 0].plot(steps, history['loss_history']['reconstruction_loss'], label='Reconstruction', alpha=0.7)
        axes[0, 0].plot(steps, history['loss_history']['sparsity_loss'], label='Sparsity', alpha=0.7)
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Components')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # Sparsity evolution
        axes[0, 1].plot(steps, history['metrics_history']['sparsity_l0'], alpha=0.7)
        axes[0, 1].axhline(y=self.sae.cfg.target_sparsity, color='red', linestyle='--', label='Target')
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Fraction Active Features')
        axes[0, 1].set_title('Sparsity Evolution')
        axes[0, 1].legend()
        
        # Sparsity coefficient
        axes[0, 2].plot(steps, history['metrics_history']['sparsity_coeff'], alpha=0.7)
        axes[0, 2].set_xlabel('Training Steps')
        axes[0, 2].set_ylabel('Sparsity Coefficient lambda')
        axes[0, 2].set_title('Sparsity Coefficient Annealing')
        axes[0, 2].set_yscale('log')
        
        # Reconstruction quality
        axes[1, 0].plot(steps, history['metrics_history']['cosine_similarity'], alpha=0.7)
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Cosine Similarity')
        axes[1, 0].set_title('Reconstruction Quality')
        
        # Feature usage
        axes[1, 1].plot(steps, history['metrics_history']['feature_usage'], alpha=0.7)
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Fraction of Features Used')
        axes[1, 1].set_title('Feature Usage')
        
        # Dead neurons over time
        if history['dead_neuron_history']:
            dead_steps = [x['step'] for x in history['dead_neuron_history']]
            dead_fractions = [x['dead_fraction'] for x in history['dead_neuron_history']]
            axes[1, 2].plot(dead_steps, dead_fractions, alpha=0.7)
            
            # Mark resampling events
            if history['resample_history']:
                resample_steps = [x['step'] for x in history['resample_history']]
                resample_fractions = [x['dead_fraction'] for x in history['resample_history']]
                axes[1, 2].scatter(resample_steps, resample_fractions, color='red', 
                                 label='Resampled', s=50, alpha=0.8)
                axes[1, 2].legend()
        
        axes[1, 2].set_xlabel('Training Steps')
        axes[1, 2].set_ylabel('Fraction Dead Neurons')
        axes[1, 2].set_title('Dead Neuron Evolution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def plot_feature_statistics(self, save_path: Optional[str] = None):
        """
        Plot feature activation statistics
        
        Args:
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Feature Activation Statistics', fontsize=16)
        
        with torch.no_grad():
            activation_freq = (self.features > 0).float().mean(dim=0).cpu().numpy()
            mean_activations = self.features.mean(dim=0).cpu().numpy()
            max_activations = self.features.max(dim=0).values.cpu().numpy()
            decoder_norms = torch.norm(self.sae.W_dec.data, dim=0).cpu().numpy()
        
        # Activation frequency histogram
        axes[0, 0].hist(activation_freq, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=self.sae.cfg.dead_neuron_threshold, color='red', 
                          linestyle='--', label='Dead Threshold')
        axes[0, 0].set_xlabel('Activation Frequency')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].set_title('Feature Activation Frequencies')
        axes[0, 0].legend()
        
        # Mean activation histogram
        axes[0, 1].hist(mean_activations, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Mean Activation')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].set_title('Mean Feature Activations')
        
        # Max activation vs frequency scatter
        axes[1, 0].scatter(activation_freq, max_activations, alpha=0.5, s=1)
        axes[1, 0].set_xlabel('Activation Frequency')
        axes[1, 0].set_ylabel('Max Activation')
        axes[1, 0].set_title('Max Activation vs Frequency')
        
        # Decoder weight norms
        axes[1, 1].hist(decoder_norms, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Decoder Weight Norm')
        axes[1, 1].set_ylabel('Number of Features')
        axes[1, 1].set_title('Decoder Weight Norms')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature statistics saved to {save_path}")
        
        plt.show()
    
    def generate_feature_report(self, n_top_features: int = 20, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive feature analysis report
        
        Args:
            n_top_features: Number of top features to analyze in detail
            save_dir: Directory to save analysis outputs
            
        Returns:
            Feature analysis report
        """
        print("Generating feature analysis report...")
        
        # Basic statistics
        stats = self.analyze_feature_statistics()
        
        # Feature correlations
        print("Computing feature correlations...")
        correlation_matrix = self.compute_feature_correlations()
        
        # Identify most/least active features
        activation_freq = (self.features > 0).float().mean(dim=0)
        most_active_features = torch.topk(activation_freq, n_top_features).indices
        least_active_features = torch.topk(-activation_freq, n_top_features).indices
        
        # Analyze top features
        print("Analyzing top features...")
        top_feature_analysis = {}
        for i, feature_idx in enumerate(most_active_features[:10]):  # Detailed analysis for top 10
            analysis = self.get_max_activating_examples(feature_idx.item())
            top_feature_analysis[f"feature_{feature_idx.item()}"] = analysis
        
        report = {
            'statistics': stats,
            'top_active_features': most_active_features.cpu().numpy().tolist(),
            'least_active_features': least_active_features.cpu().numpy().tolist(),
            'feature_analysis': top_feature_analysis,
            'correlation_stats': {
                'mean_correlation': correlation_matrix.mean().item(),
                'max_correlation': correlation_matrix.max().item(),
                'min_correlation': correlation_matrix.min().item()
            }
        }
        
        # Save report
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save JSON report
            report_path = os.path.join(save_dir, "feature_analysis_report.json")
            with open(report_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_report = json.dumps(report, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
                f.write(json_report)
            
            # Save correlation matrix
            corr_path = os.path.join(save_dir, "feature_correlations.pt")
            torch.save(correlation_matrix, corr_path)
            
            print(f"Feature analysis report saved to {save_dir}")
        
        return report