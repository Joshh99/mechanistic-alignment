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
from tqdm import tqdm
import logging

# Import our custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sae import MiniSAE, SAEConfig, ActivationCollector
from sae.training import SAETrainer, SAEAnalyzer
from transformer.attention import HookedTransformerWrapper
from .alignment_dataset import AlignmentDatasetGenerator, AlignmentExample, BehaviorType, PromptType


@dataclass
class AlignmentSAEConfig:
    """Configuration for alignment-focused SAE training"""
    
    # Dataset parameters
    n_examples: int = 3000                    # Number of alignment examples
    target_activations: int = 75000           # Target number of activation vectors
    validation_split: float = 0.1            # Fraction for validation
    
    # Model parameters  
    d_in: int = 768                          # Transformer dimension
    expansion_factor: int = 4                # SAE expansion factor
    hook_name: str = "blocks.2.hook_resid_post"  # Target layer for alignment features
    
    # Training parameters (conservative for alignment discovery)
    batch_size: int = 2048                   # Large batches for stability
    learning_rate: float = 1e-4              # Conservative learning rate
    sparsity_coeff: float = 1e-3             # Initial sparsity coefficient
    max_sparsity_coeff: float = 5e-2         # Maximum sparsity coefficient
    target_sparsity: float = 0.08            # Target 8% active features (5-15% range)
    
    # Training schedule
    n_epochs: int = 30                       # Conservative epoch count
    warmup_epochs: int = 8                   # Longer warmup for stability
    patience: int = 5                        # Early stopping patience
    min_improvement: float = 1e-5            # Minimum improvement for early stopping
    
    # Quality thresholds
    max_reconstruction_error: float = 0.15   # <15% reconstruction error
    min_sparsity: float = 0.05              # Minimum 5% sparsity
    max_sparsity: float = 0.15              # Maximum 15% sparsity
    max_dead_neurons: float = 0.20          # <20% dead neurons
    
    # Logging and checkpointing
    log_interval: int = 25                   # Log every N steps
    checkpoint_interval: int = 5             # Save every N epochs
    metrics_window: int = 100                # Rolling metrics window
    
    # Device and paths
    device: str = "cpu"                      # Training device
    save_dir: str = "alignment_sae_training" # Output directory
    dataset_path: Optional[str] = None       # Pre-generated dataset path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AlignmentSAEConfig':
        """Load from dictionary"""
        return cls(**config_dict)


class AlignmentActivationCollector(ActivationCollector):
    """
    Specialized activation collector for alignment features
    
    Extends base collector with:
    - Behavioral labeling preservation
    - Statistical validation during collection
    - Quality assurance metrics
    """
    
    def __init__(self, model: HookedTransformerWrapper, hook_name: str, 
                 max_samples: int = 100000, device: str = "cpu"):
        super().__init__(model, hook_name, max_samples)
        self.device = device
        self.behavior_labels = []  # Store behavior labels with activations
        self.example_metadata = []  # Store example metadata
        self.collection_stats = defaultdict(list)
        
    def collect_from_alignment_examples(self, examples: List[AlignmentExample],
                                      response_type: str = "both") -> torch.Tensor:
        """
        Collect activations from alignment examples
        
        Args:
            examples: List of alignment examples
            response_type: "sycophantic", "honest", or "both"
            
        Returns:
            Collected activation tensor
        """
        print(f"Collecting activations from {len(examples)} alignment examples...")
        print(f"Target hook: {self.hook_name}")
        print(f"Response type: {response_type}")
        
        # Register collection hook
        hook_points = self.model._get_hook_points()
        if self.hook_name not in hook_points:
            raise ValueError(f"Hook {self.hook_name} not found in model")
        
        hook_point = hook_points[self.hook_name]
        hook_point.add_hook(self._labeled_collection_hook)
        
        try:
            batch_texts = []
            batch_labels = []
            batch_metadata = []
            
            for i, example in enumerate(tqdm(examples, desc="Processing examples")):
                # Get text(s) to process based on response_type
                texts_to_process = []
                if response_type == "sycophantic":
                    texts_to_process = [example.prompt + " " + example.sycophantic_response]
                elif response_type == "honest":
                    texts_to_process = [example.prompt + " " + example.honest_response]
                elif response_type == "both":
                    texts_to_process = [
                        example.prompt + " " + example.sycophantic_response,
                        example.prompt + " " + example.honest_response
                    ]
                
                for j, text in enumerate(texts_to_process):
                    # Convert text to tokens (simplified - in practice, use proper tokenizer)
                    tokens = self._text_to_tokens(text)
                    batch_texts.append(tokens)
                    
                    # Create labels for this response
                    labels = example.behavior_labels.copy()
                    labels['response_type'] = response_type if response_type != "both" else ("sycophantic" if j == 0 else "honest")
                    batch_labels.append(labels)
                    
                    # Store metadata
                    metadata = {
                        'example_idx': i,
                        'prompt_type': example.prompt_type.value,
                        'response_idx': j,
                        **example.metadata
                    }
                    batch_metadata.append(metadata)
                
                # Process in batches to avoid memory issues
                if len(batch_texts) >= 32 or i == len(examples) - 1:
                    self._process_batch(batch_texts, batch_labels, batch_metadata)
                    batch_texts = []
                    batch_labels = []
                    batch_metadata = []
                
                if self.collected_samples >= self.max_samples:
                    print(f"Reached maximum samples: {self.max_samples}")
                    break
            
            # Combine all collected activations
            if self.activations:
                all_activations = torch.cat(self.activations, dim=0)
                
                # Validate collection quality
                self._validate_collection_quality(all_activations)
                
                print(f"Collection complete: {all_activations.shape[0]} activation vectors")
                return all_activations
            else:
                raise ValueError("No activations collected")
                
        finally:
            # Clean up hook
            hook_point.remove_hooks()
    
    def _labeled_collection_hook(self, activation: torch.Tensor, hook_name: str) -> torch.Tensor:
        """Enhanced collection hook that preserves labels"""
        if self.collected_samples < self.max_samples:
            # Flatten [batch, seq_len, d_model] -> [batch*seq_len, d_model]
            flat_activation = activation.reshape(-1, activation.shape[-1])
            
            # Store on CPU to save GPU memory
            self.activations.append(flat_activation.detach().cpu())
            self.collected_samples += flat_activation.shape[0]
            
            # Compute and store collection statistics
            self._update_collection_stats(flat_activation)
        
        return activation
    
    def _process_batch(self, batch_texts: List[torch.Tensor], batch_labels: List[Dict],
                      batch_metadata: List[Dict]):
        """Process a batch of texts through the model"""
        if not batch_texts:
            return
        
        # Stack texts into batch tensor
        max_len = max(len(text) for text in batch_texts)
        batch_tensor = torch.zeros(len(batch_texts), max_len, dtype=torch.long)
        
        for i, text in enumerate(batch_texts):
            batch_tensor[i, :len(text)] = text
        
        batch_tensor = batch_tensor.to(self.device)
        
        # Store labels and metadata for this batch
        self.behavior_labels.extend(batch_labels)
        self.example_metadata.extend(batch_metadata)
        
        # Forward pass to trigger hook
        with torch.no_grad():
            _ = self.model(batch_tensor)
    
    def _text_to_tokens(self, text: str, max_length: int = 128) -> torch.Tensor:
        """
        Convert text to token tensor (simplified version)
        In practice, use proper tokenizer
        """
        # Simplified tokenization - just use character codes modulo vocab size
        tokens = [ord(c) % 1000 for c in text.lower()[:max_length]]
        return torch.tensor(tokens, dtype=torch.long)
    
    def _update_collection_stats(self, activation: torch.Tensor):
        """Update collection statistics"""
        with torch.no_grad():
            self.collection_stats['mean'].append(activation.mean().item())
            self.collection_stats['std'].append(activation.std().item())
            self.collection_stats['min'].append(activation.min().item())
            self.collection_stats['max'].append(activation.max().item())
            self.collection_stats['norm'].append(torch.norm(activation, dim=1).mean().item())
    
    def _validate_collection_quality(self, activations: torch.Tensor):
        """Validate quality of collected activations"""
        print("\nValidating activation collection quality...")
        
        # Basic statistics
        mean_val = activations.mean().item()
        std_val = activations.std().item()
        min_val = activations.min().item()
        max_val = activations.max().item()
        
        print(f"Activation statistics:")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std: {std_val:.4f}")
        print(f"  Min: {min_val:.4f}")
        print(f"  Max: {max_val:.4f}")
        
        # Quality checks
        warnings_issued = []
        
        if std_val < 0.01:
            warnings_issued.append("Very low activation variance - may indicate poor feature diversity")
        
        if std_val > 1.0:
            warnings_issued.append("Very high activation variance - may indicate instability")
        
        if abs(mean_val) > 0.5:
            warnings_issued.append("High mean activation - may indicate bias")
        
        # Check for NaN or inf values
        if torch.isnan(activations).any():
            warnings_issued.append("NaN values detected in activations")
        
        if torch.isinf(activations).any():
            warnings_issued.append("Infinite values detected in activations")
        
        # Distribution analysis
        activation_norms = torch.norm(activations, dim=1)
        if (activation_norms < 1e-6).sum().item() > activations.shape[0] * 0.01:
            warnings_issued.append("Many near-zero activation vectors detected")
        
        if warnings_issued:
            print("\nWarnings:")
            for warning in warnings_issued:
                print(f"  - {warning}")
        else:
            print("  All quality checks passed!")
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of collection process"""
        return {
            'total_samples': self.collected_samples,
            'behavior_labels_collected': len(self.behavior_labels),
            'metadata_collected': len(self.example_metadata),
            'collection_stats': {k: np.mean(v) for k, v in self.collection_stats.items()},
            'hook_name': self.hook_name,
        }


class AlignmentSAETrainer:
    """
    Complete SAE training pipeline for alignment feature discovery
    
    Features:
    - Conservative training with quality monitoring
    - Real-time metrics tracking and validation
    - Automatic convergence detection and early stopping
    - Comprehensive logging and checkpointing
    """
    
    def __init__(self, config: AlignmentSAEConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create output directory
        os.makedirs(config.save_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.model = None
        self.sae = None
        self.trainer = None
        self.dataset_examples = None
        self.activations = None
        
        # Training state
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = defaultdict(list)
        self.quality_metrics = defaultdict(list)
        
        self.logger.info(f"AlignmentSAETrainer initialized with config: {config}")
    
    def _setup_logging(self):
        """Setup logging system"""
        log_path = os.path.join(self.config.save_dir, 'training.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def initialize_model(self, model_config: Dict[str, Any]) -> HookedTransformerWrapper:
        """Initialize the transformer model"""
        self.logger.info("Initializing HookedTransformerWrapper...")
        
        self.model = HookedTransformerWrapper(model_config)
        self.model.to(self.device)
        
        # Initialize weights properly
        with torch.no_grad():
            for param in self.model.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    param.fill_(0.01)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model initialized with {n_params:,} parameters")
        
        return self.model
    
    def create_alignment_dataset(self, force_regenerate: bool = False) -> List[AlignmentExample]:
        """Create or load alignment dataset"""
        dataset_path = self.config.dataset_path or os.path.join(self.config.save_dir, 'alignment_dataset.json')
        
        if os.path.exists(dataset_path) and not force_regenerate:
            self.logger.info(f"Loading existing dataset from {dataset_path}")
            examples, metadata = AlignmentDatasetGenerator.load_dataset(dataset_path)
            self.dataset_examples = examples
        else:
            self.logger.info(f"Generating new alignment dataset with {self.config.n_examples} examples")
            generator = AlignmentDatasetGenerator(seed=42)
            examples = generator.generate_dataset(n_examples=self.config.n_examples)
            generator.save_dataset(examples, dataset_path)
            self.dataset_examples = examples
        
        # Analyze dataset composition
        self._analyze_dataset_composition()
        
        return self.dataset_examples
    
    def _analyze_dataset_composition(self):
        """Analyze and log dataset composition"""
        if not self.dataset_examples:
            return
        
        self.logger.info("Dataset composition analysis:")
        
        # Prompt type distribution
        prompt_type_counts = defaultdict(int)
        for example in self.dataset_examples:
            prompt_type_counts[example.prompt_type.value] += 1
        
        self.logger.info("Prompt type distribution:")
        for prompt_type, count in prompt_type_counts.items():
            percentage = count / len(self.dataset_examples) * 100
            self.logger.info(f"  {prompt_type}: {count} ({percentage:.1f}%)")
        
        # Behavior label statistics
        behavior_stats = defaultdict(list)
        for example in self.dataset_examples:
            for behavior, score in example.behavior_labels.items():
                behavior_stats[behavior].append(score)
        
        self.logger.info("Behavior label statistics:")
        for behavior, scores in behavior_stats.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            self.logger.info(f"  {behavior}: mean={mean_score:.3f}, std={std_score:.3f}")
    
    def collect_activations(self) -> torch.Tensor:
        """Collect activations from alignment examples"""
        if not self.model:
            raise ValueError("Model must be initialized before collecting activations")
        
        if not self.dataset_examples:
            raise ValueError("Dataset must be created before collecting activations")
        
        self.logger.info("Starting activation collection...")
        
        # Initialize collector
        collector = AlignmentActivationCollector(
            model=self.model,
            hook_name=self.config.hook_name,
            max_samples=self.config.target_activations,
            device=self.device
        )
        
        # Collect activations from both sycophantic and honest responses
        activations = collector.collect_from_alignment_examples(
            examples=self.dataset_examples,
            response_type="both"
        )
        
        # Store activations and collection metadata
        self.activations = activations
        collection_summary = collector.get_collection_summary()
        
        # Save collection summary
        summary_path = os.path.join(self.config.save_dir, 'collection_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(collection_summary, f, indent=2)
        
        self.logger.info(f"Activation collection complete: {activations.shape}")
        self.logger.info(f"Collection summary saved to {summary_path}")
        
        return activations
    
    def initialize_sae(self) -> MiniSAE:
        """Initialize SAE with alignment-focused configuration"""
        if self.activations is None:
            raise ValueError("Activations must be collected before initializing SAE")
        
        self.logger.info("Initializing SAE for alignment feature discovery...")
        
        # Get actual activation dimension from collected data
        actual_d_in = self.activations.shape[1]
        self.logger.info(f"Using actual activation dimension: {actual_d_in}")
        
        # Create SAE config
        sae_config = SAEConfig(
            d_in=actual_d_in,  # Use actual dimension from activations
            expansion_factor=self.config.expansion_factor,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            sparsity_coeff=self.config.sparsity_coeff,
            max_sparsity_coeff=self.config.max_sparsity_coeff,
            target_sparsity=self.config.target_sparsity,
            n_epochs=self.config.n_epochs,
            warmup_epochs=self.config.warmup_epochs,
            log_interval=self.config.log_interval,
            device=self.config.device,
            hook_name=self.config.hook_name
        )
        
        # Initialize SAE
        self.sae = MiniSAE(sae_config)
        self.sae.to(self.device)
        
        n_params = sum(p.numel() for p in self.sae.parameters())
        self.logger.info(f"SAE initialized: {sae_config.d_in} -> {sae_config.d_sae} -> {sae_config.d_in}")
        self.logger.info(f"SAE parameters: {n_params:,}")
        
        return self.sae
    
    def train_sae(self) -> Dict[str, Any]:
        """Train SAE with quality monitoring and early stopping"""
        if self.sae is None or self.activations is None:
            raise ValueError("SAE and activations must be initialized before training")
        
        self.logger.info("Starting SAE training with quality monitoring...")
        
        # Initialize base trainer
        self.trainer = SAETrainer(self.sae, self.sae.cfg)
        
        # Split data for training and validation
        n_train = int(self.activations.shape[0] * (1 - self.config.validation_split))
        train_activations = self.activations[:n_train]
        val_activations = self.activations[n_train:]
        
        self.logger.info(f"Training split: {train_activations.shape[0]} train, {val_activations.shape[0]} validation")
        
        # Create data loaders
        train_dataset = TensorDataset(train_activations)
        val_dataset = TensorDataset(val_activations)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=False)
        
        # Training loop with enhanced monitoring
        for epoch in range(self.config.n_epochs):
            epoch_start = time.time()
            
            # Train epoch
            train_metrics = self._train_epoch_with_monitoring(train_loader, epoch)
            
            # Validate epoch
            val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Quality assurance checks
            quality_passed = self._quality_assurance_check(train_metrics, val_metrics, epoch)
            
            # Early stopping check
            current_loss = val_metrics['loss_total_loss']
            if current_loss < self.best_loss - self.config.min_improvement:
                self.best_loss = current_loss
                self.epochs_without_improvement = 0
                
                # Save best model
                best_path = os.path.join(self.config.save_dir, 'sae_best.pt')
                self.sae.save_checkpoint(best_path, self.trainer.optimizer.state_dict())
                self.logger.info(f"New best model saved: loss={current_loss:.6f}")
            else:
                self.epochs_without_improvement += 1
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch+1}/{self.config.n_epochs} ({epoch_time:.1f}s)")
            self.logger.info(f"Train Loss: {train_metrics['loss_total_loss']:.6f}, Val Loss: {val_metrics['loss_total_loss']:.6f}")
            self.logger.info(f"Sparsity: {train_metrics['metric_sparsity_l0']:.3f}, Quality: {'PASS' if quality_passed else 'FAIL'}")
            
            # Checkpointing
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                checkpoint_path = os.path.join(self.config.save_dir, f'sae_epoch_{epoch+1}.pt')
                self.sae.save_checkpoint(checkpoint_path, self.trainer.optimizer.state_dict())
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.patience:
                self.logger.info(f"Early stopping: no improvement for {self.config.patience} epochs")
                break
            
            # Quality failure stopping
            if not quality_passed and epoch > self.config.warmup_epochs:
                self.logger.warning("Training stopped due to quality assurance failure")
                break
        
        # Final model save
        final_path = os.path.join(self.config.save_dir, 'sae_final.pt')
        self.sae.save_checkpoint(final_path, self.trainer.optimizer.state_dict())
        
        # Save training history
        history_path = os.path.join(self.config.save_dir, 'training_history.json')
        self._save_training_history(history_path)
        
        self.logger.info("SAE training complete!")
        self.logger.info(f"Best validation loss: {self.best_loss:.6f}")
        
        return {
            'training_history': dict(self.training_history),
            'quality_metrics': dict(self.quality_metrics),
            'best_loss': self.best_loss,
            'final_epoch': epoch + 1
        }
    
    def _train_epoch_with_monitoring(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train epoch with enhanced monitoring"""
        self.sae.train()
        epoch_losses = defaultdict(list)
        epoch_metrics = defaultdict(list)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, (batch_activations,) in enumerate(progress_bar):
            batch_activations = batch_activations.to(self.device)
            
            # Update sparsity coefficient
            sparsity_coeff = self.trainer.compute_sparsity_coefficient(epoch, step)
            
            # Forward pass
            reconstruction, features = self.sae(batch_activations)
            
            # Compute loss
            loss_dict = self.sae.compute_loss(batch_activations, reconstruction, features, sparsity_coeff)
            
            # Backward pass
            self.trainer.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.sae.parameters(), max_norm=1.0)
            self.trainer.optimizer.step()
            
            # Normalize decoder weights
            self.sae.normalize_decoder_weights()
            
            # Update feature statistics
            self.sae.update_feature_stats(features.detach())
            
            # Compute metrics
            sparsity_metrics = self.sae.get_sparsity_metrics(features.detach())
            reconstruction_metrics = self.sae.get_reconstruction_metrics(batch_activations, reconstruction.detach())
            
            # Store metrics
            for key, value in loss_dict.items():
                epoch_losses[key].append(value.item())
            for key, value in {**sparsity_metrics, **reconstruction_metrics}.items():
                epoch_metrics[key].append(value)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.6f}",
                'recon': f"{loss_dict['reconstruction_loss'].item():.6f}",
                'sparsity': f"{sparsity_metrics['sparsity_l0']:.3f}",
                'lambda': f"{sparsity_coeff:.2e}"
            })
            
            # Real-time quality monitoring
            if step % self.config.log_interval == 0:
                self._log_realtime_metrics(epoch, step, loss_dict, sparsity_metrics, reconstruction_metrics)
        
        # Compute epoch averages
        epoch_summary = {}
        for key, values in epoch_losses.items():
            epoch_summary[f"loss_{key}"] = np.mean(values)
        for key, values in epoch_metrics.items():
            epoch_summary[f"metric_{key}"] = np.mean(values)
        
        return epoch_summary
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate epoch with comprehensive metrics"""
        self.sae.eval()
        val_losses = defaultdict(list)
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch_activations, in val_loader:
                batch_activations = batch_activations.to(self.device)
                
                reconstruction, features = self.sae(batch_activations)
                sparsity_coeff = self.trainer.current_sparsity_coeff
                
                loss_dict = self.sae.compute_loss(batch_activations, reconstruction, features, sparsity_coeff)
                sparsity_metrics = self.sae.get_sparsity_metrics(features)
                reconstruction_metrics = self.sae.get_reconstruction_metrics(batch_activations, reconstruction)
                
                for key, value in loss_dict.items():
                    val_losses[key].append(value.item())
                for key, value in {**sparsity_metrics, **reconstruction_metrics}.items():
                    val_metrics[key].append(value)
        
        # Compute validation averages
        val_summary = {}
        for key, values in val_losses.items():
            val_summary[f"loss_{key}"] = np.mean(values)
        for key, values in val_metrics.items():
            val_summary[f"metric_{key}"] = np.mean(values)
        
        return val_summary
    
    def _quality_assurance_check(self, train_metrics: Dict, val_metrics: Dict, epoch: int) -> bool:
        """Comprehensive quality assurance check"""
        quality_passed = True
        issues = []
        
        # Check reconstruction error threshold
        recon_error = val_metrics['loss_reconstruction_loss']
        if recon_error > self.config.max_reconstruction_error:
            quality_passed = False
            issues.append(f"Reconstruction error too high: {recon_error:.3f} > {self.config.max_reconstruction_error}")
        
        # Check sparsity bounds
        sparsity = val_metrics['metric_sparsity_l0']
        if sparsity < self.config.min_sparsity:
            quality_passed = False
            issues.append(f"Sparsity too low: {sparsity:.3f} < {self.config.min_sparsity}")
        elif sparsity > self.config.max_sparsity:
            quality_passed = False
            issues.append(f"Sparsity too high: {sparsity:.3f} > {self.config.max_sparsity}")
        
        # Check dead neurons
        dead_mask = self.sae.get_dead_neurons()
        dead_fraction = dead_mask.float().mean().item()
        if dead_fraction > self.config.max_dead_neurons:
            quality_passed = False
            issues.append(f"Too many dead neurons: {dead_fraction:.1%} > {self.config.max_dead_neurons:.1%}")
        
        # Check for NaN/inf values
        if torch.isnan(torch.tensor(recon_error)) or torch.isinf(torch.tensor(recon_error)):
            quality_passed = False
            issues.append("NaN or infinite values detected in loss")
        
        # Store quality metrics
        self.quality_metrics['reconstruction_error'].append(recon_error)
        self.quality_metrics['sparsity'].append(sparsity)
        self.quality_metrics['dead_fraction'].append(dead_fraction)
        self.quality_metrics['quality_passed'].append(quality_passed)
        
        if not quality_passed and issues:
            self.logger.warning(f"Quality check failed at epoch {epoch+1}:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
        
        return quality_passed
    
    def _log_realtime_metrics(self, epoch: int, step: int, loss_dict: Dict, 
                            sparsity_metrics: Dict, reconstruction_metrics: Dict):
        """Log real-time training metrics"""
        metrics = {
            'epoch': epoch,
            'step': step,
            'loss_total': loss_dict['total_loss'].item(),
            'loss_recon': loss_dict['reconstruction_loss'].item(),
            'loss_sparsity': loss_dict['sparsity_loss'].item(),
            'sparsity_l0': sparsity_metrics['sparsity_l0'],
            'cosine_sim': reconstruction_metrics['cosine_similarity'],
            'feature_usage': sparsity_metrics['feature_usage'],
        }
        
        for key, value in metrics.items():
            self.training_history[key].append(value)
    
    def _save_training_history(self, path: str):
        """Save comprehensive training history"""
        history = {
            'config': self.config.to_dict(),
            'training_metrics': {k: v for k, v in self.training_history.items()},
            'quality_metrics': {k: v for k, v in self.quality_metrics.items()},
            'model_info': {
                'best_loss': self.best_loss,
                'total_epochs': len(self.quality_metrics['quality_passed']),
                'early_stopped': self.epochs_without_improvement >= self.config.patience
            }
        }
        
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Training history saved to {path}")
    
    def run_complete_pipeline(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete SAE training pipeline
        
        Args:
            model_config: Configuration for transformer model
            
        Returns:
            Complete training results
        """
        self.logger.info("Starting complete SAE training pipeline for alignment feature discovery")
        
        pipeline_start = time.time()
        
        try:
            # Step 1: Initialize model
            self.initialize_model(model_config)
            
            # Step 2: Create/load alignment dataset
            self.create_alignment_dataset()
            
            # Step 3: Collect activations
            self.collect_activations()
            
            # Step 4: Initialize SAE
            self.initialize_sae()
            
            # Step 5: Train SAE
            training_results = self.train_sae()
            
            # Step 6: Generate final analysis
            analysis_results = self._generate_final_analysis()
            
            pipeline_time = time.time() - pipeline_start
            
            complete_results = {
                'pipeline_success': True,
                'pipeline_time': pipeline_time,
                'model_config': model_config,
                'training_config': self.config.to_dict(),
                'training_results': training_results,
                'analysis_results': analysis_results,
                'output_directory': self.config.save_dir
            }
            
            # Save complete results
            results_path = os.path.join(self.config.save_dir, 'complete_results.json')
            with open(results_path, 'w') as f:
                json.dump(complete_results, f, indent=2, default=str)
            
            self.logger.info(f"Complete pipeline finished successfully in {pipeline_time:.1f}s")
            self.logger.info(f"Results saved to {self.config.save_dir}")
            
            return complete_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _generate_final_analysis(self) -> Dict[str, Any]:
        """Generate final analysis of trained SAE"""
        if self.sae is None or self.activations is None:
            return {}
        
        self.logger.info("Generating final SAE analysis...")
        
        # Initialize analyzer
        analyzer = SAEAnalyzer(self.sae, self.activations)
        
        # Generate comprehensive analysis
        feature_report = analyzer.generate_feature_report(
            n_top_features=20,
            save_dir=os.path.join(self.config.save_dir, 'analysis')
        )
        
        # Generate plots if possible
        try:
            if self.training_history:
                analyzer.plot_training_curves(
                    {'metrics_history': dict(self.training_history)},
                    save_path=os.path.join(self.config.save_dir, 'training_curves.png')
                )
        except Exception as e:
            self.logger.warning(f"Could not generate training plots: {e}")
        
        return feature_report


# Configuration factory function
def create_alignment_sae_config(**kwargs) -> AlignmentSAEConfig:
    """Create alignment SAE configuration with overrides"""
    return AlignmentSAEConfig(**kwargs)


if __name__ == "__main__":
    # Example usage
    config = create_alignment_sae_config(
        n_examples=1000,
        target_activations=25000,
        n_epochs=15,
        device="cpu"
    )
    
    model_config = {
        "vocab_size": 50257,
        "context_length": 512,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 6,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
    
    trainer = AlignmentSAETrainer(config)
    results = trainer.run_complete_pipeline(model_config)
    
    print(f"Training completed successfully!")
    print(f"Results saved to: {results['output_directory']}")
    print(f"Best validation loss: {results['training_results']['best_loss']:.6f}")