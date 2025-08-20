import os
import time
import json
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
import pandas as pd
from tqdm import tqdm
import threading
import queue
from datetime import datetime


@dataclass
class TrainingQualityThresholds:
    """Quality thresholds for training monitoring"""
    
    # Reconstruction quality
    min_cosine_similarity: float = 0.9
    min_explained_variance: float = 0.80
    max_reconstruction_error: float = 0.15
    
    # Sparsity requirements
    min_sparsity_l0: float = 0.05      # 5% minimum activation
    max_sparsity_l0: float = 0.15      # 15% maximum activation
    target_sparsity_l0: float = 0.08   # 8% target activation
    
    # Dead neuron management
    max_dead_fraction: float = 0.20    # <20% dead neurons
    dead_neuron_threshold: float = 1e-6
    resample_trigger_fraction: float = 0.25  # Resample when >25% dead
    
    # Training convergence
    convergence_window: int = 100      # Steps to check for convergence
    min_loss_improvement: float = 1e-5 # Minimum improvement for convergence
    gradient_norm_threshold: float = 10.0  # Max gradient norm
    
    # Feature quality
    min_feature_coherence: float = 0.3
    min_semantic_consistency: float = 0.4
    max_feature_redundancy: float = 0.8


@dataclass
class FeatureQualityMetrics:
    """Metrics for evaluating individual feature quality"""
    
    feature_idx: int
    activation_frequency: float
    mean_activation: float
    max_activation: float
    coherence_score: float
    semantic_consistency: float
    redundancy_score: float
    interpretability_score: float
    confidence_score: float
    quality_passed: bool


class RealTimeMetricsDashboard:
    """
    Real-time training metrics dashboard with live plotting
    
    Tracks and visualizes training progress with quality indicators
    """
    
    def __init__(self, thresholds: TrainingQualityThresholds, 
                 update_interval: int = 10, max_history: int = 1000):
        self.thresholds = thresholds
        self.update_interval = update_interval
        self.max_history = max_history
        
        # Metric storage with bounded history
        self.metrics = {
            'step': deque(maxlen=max_history),
            'total_loss': deque(maxlen=max_history),
            'reconstruction_loss': deque(maxlen=max_history),
            'sparsity_loss': deque(maxlen=max_history),
            'cosine_similarity': deque(maxlen=max_history),
            'explained_variance': deque(maxlen=max_history),
            'sparsity_l0': deque(maxlen=max_history),
            'dead_fraction': deque(maxlen=max_history),
            'gradient_norm': deque(maxlen=max_history),
            'learning_rate': deque(maxlen=max_history),
            'sparsity_coeff': deque(maxlen=max_history)
        }
        
        # Quality status tracking
        self.quality_status = {
            'reconstruction_quality': True,
            'sparsity_range': True,
            'dead_neurons': True,
            'gradient_stability': True,
            'overall': True
        }
        
        # Convergence tracking
        self.convergence_tracker = deque(maxlen=thresholds.convergence_window)
        self.last_plot_time = 0
        
    def update_metrics(self, step: int, metrics_dict: Dict[str, float]):
        """Update metrics with new training step data"""
        self.metrics['step'].append(step)
        
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        # Update quality status
        self._update_quality_status()
        
        # Check for convergence
        self._check_convergence()
        
        # Plot if enough time has passed
        current_time = time.time()
        if current_time - self.last_plot_time > self.update_interval:
            self._update_plots()
            self.last_plot_time = current_time
    
    def _update_quality_status(self):
        """Update quality status based on latest metrics"""
        if not self.metrics['cosine_similarity']:
            return
        
        latest = {key: values[-1] if values else 0 for key, values in self.metrics.items()}
        
        # Check reconstruction quality
        recon_quality = (
            latest['cosine_similarity'] >= self.thresholds.min_cosine_similarity and
            latest['explained_variance'] >= self.thresholds.min_explained_variance
        )
        self.quality_status['reconstruction_quality'] = recon_quality
        
        # Check sparsity range
        sparsity_ok = (
            self.thresholds.min_sparsity_l0 <= latest['sparsity_l0'] <= self.thresholds.max_sparsity_l0
        )
        self.quality_status['sparsity_range'] = sparsity_ok
        
        # Check dead neurons
        dead_ok = latest['dead_fraction'] <= self.thresholds.max_dead_fraction
        self.quality_status['dead_neurons'] = dead_ok
        
        # Check gradient stability
        grad_ok = latest['gradient_norm'] <= self.thresholds.gradient_norm_threshold
        self.quality_status['gradient_stability'] = grad_ok
        
        # Overall status
        self.quality_status['overall'] = all([
            recon_quality, sparsity_ok, dead_ok, grad_ok
        ])
    
    def _check_convergence(self):
        """Check if training has converged"""
        if len(self.metrics['total_loss']) < self.thresholds.convergence_window:
            return False
        
        recent_losses = list(self.metrics['total_loss'])[-self.thresholds.convergence_window:]
        loss_improvement = recent_losses[0] - recent_losses[-1]
        
        converged = loss_improvement < self.thresholds.min_loss_improvement
        return converged
    
    def _update_plots(self):
        """Update real-time plots (non-blocking)"""
        try:
            if len(self.metrics['step']) < 2:
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'SAE Training Dashboard - Step {self.metrics["step"][-1]}', fontsize=16)
            
            steps = list(self.metrics['step'])
            
            # Loss curves
            axes[0, 0].plot(steps, list(self.metrics['total_loss']), 'b-', alpha=0.7, label='Total')
            axes[0, 0].plot(steps, list(self.metrics['reconstruction_loss']), 'r-', alpha=0.7, label='Reconstruction')
            axes[0, 0].plot(steps, list(self.metrics['sparsity_loss']), 'g-', alpha=0.7, label='Sparsity')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Losses')
            axes[0, 0].legend()
            axes[0, 0].set_yscale('log')
            
            # Reconstruction quality
            axes[0, 1].plot(steps, list(self.metrics['cosine_similarity']), 'b-', alpha=0.7, label='Cosine Sim')
            axes[0, 1].plot(steps, list(self.metrics['explained_variance']), 'r-', alpha=0.7, label='Explained Var')
            axes[0, 1].axhline(y=self.thresholds.min_cosine_similarity, color='b', linestyle='--', alpha=0.5)
            axes[0, 1].axhline(y=self.thresholds.min_explained_variance, color='r', linestyle='--', alpha=0.5)
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].set_title('Reconstruction Quality')
            axes[0, 1].legend()
            
            # Sparsity monitoring
            axes[0, 2].plot(steps, list(self.metrics['sparsity_l0']), 'g-', alpha=0.7)
            axes[0, 2].axhline(y=self.thresholds.target_sparsity_l0, color='g', linestyle='-', alpha=0.8, label='Target')
            axes[0, 2].axhline(y=self.thresholds.min_sparsity_l0, color='r', linestyle='--', alpha=0.5, label='Min')
            axes[0, 2].axhline(y=self.thresholds.max_sparsity_l0, color='r', linestyle='--', alpha=0.5, label='Max')
            axes[0, 2].set_ylabel('L0 Sparsity')
            axes[0, 2].set_title('Sparsity Monitoring')
            axes[0, 2].legend()
            
            # Dead neurons
            axes[1, 0].plot(steps, list(self.metrics['dead_fraction']), 'purple', alpha=0.7)
            axes[1, 0].axhline(y=self.thresholds.max_dead_fraction, color='r', linestyle='--', alpha=0.5, label='Threshold')
            axes[1, 0].axhline(y=self.thresholds.resample_trigger_fraction, color='orange', linestyle='--', alpha=0.5, label='Resample')
            axes[1, 0].set_ylabel('Dead Fraction')
            axes[1, 0].set_title('Dead Neuron Monitoring')
            axes[1, 0].legend()
            
            # Training diagnostics
            axes[1, 1].plot(steps, list(self.metrics['gradient_norm']), 'orange', alpha=0.7, label='Grad Norm')
            axes[1, 1].axhline(y=self.thresholds.gradient_norm_threshold, color='r', linestyle='--', alpha=0.5, label='Threshold')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].set_title('Training Diagnostics')
            axes[1, 1].legend()
            axes[1, 1].set_yscale('log')
            
            # Training coefficients
            axes[1, 2].plot(steps, list(self.metrics['sparsity_coeff']), 'brown', alpha=0.7, label='Sparsity Coeff')
            if self.metrics['learning_rate']:
                axes[1, 2].plot(steps, list(self.metrics['learning_rate']), 'navy', alpha=0.7, label='Learning Rate')
            axes[1, 2].set_ylabel('Coefficient Value')
            axes[1, 2].set_title('Training Parameters')
            axes[1, 2].legend()
            axes[1, 2].set_yscale('log')
            
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01)  # Brief pause to update display
            
        except Exception as e:
            # Silently handle plotting errors to not interrupt training
            pass
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Generate current quality status report"""
        if not self.metrics['step']:
            return {'status': 'no_data', 'metrics': {}}
        
        latest_metrics = {key: values[-1] if values else 0 for key, values in self.metrics.items()}
        
        return {
            'status': 'pass' if self.quality_status['overall'] else 'fail',
            'quality_checks': self.quality_status.copy(),
            'latest_metrics': latest_metrics,
            'convergence_status': self._check_convergence(),
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate training recommendations based on current status"""
        recommendations = []
        
        if not self.quality_status['reconstruction_quality']:
            recommendations.append("Increase reconstruction weight or decrease sparsity coefficient")
        
        if not self.quality_status['sparsity_range']:
            latest_sparsity = self.metrics['sparsity_l0'][-1] if self.metrics['sparsity_l0'] else 0
            if latest_sparsity < self.thresholds.min_sparsity_l0:
                recommendations.append("Increase sparsity coefficient to reduce over-activation")
            elif latest_sparsity > self.thresholds.max_sparsity_l0:
                recommendations.append("Decrease sparsity coefficient to increase activation")
        
        if not self.quality_status['dead_neurons']:
            recommendations.append("Trigger dead neuron resampling")
        
        if not self.quality_status['gradient_stability']:
            recommendations.append("Reduce learning rate or add gradient clipping")
        
        return recommendations


class SAETrainingMonitor:
    """
    Comprehensive SAE training monitor with automated quality assurance
    
    Provides real-time monitoring, automated interventions, and quality reporting
    """
    
    def __init__(self, sae_model, thresholds: Optional[TrainingQualityThresholds] = None):
        self.sae = sae_model
        self.thresholds = thresholds or TrainingQualityThresholds()
        
        # Initialize dashboard
        self.dashboard = RealTimeMetricsDashboard(self.thresholds)
        
        # Training state
        self.step_count = 0
        self.resampling_history = []
        self.intervention_history = []
        self.quality_violations = []
        
        # Automated intervention flags
        self.auto_resample_dead = True
        self.auto_adjust_sparsity = True
        self.auto_reduce_lr = True
        
    def monitor_training_step(self, step: int, optimizer, batch_data: torch.Tensor, 
                            loss_dict: Dict[str, torch.Tensor], 
                            metrics_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Monitor single training step with automated quality assurance
        
        Returns monitoring results and any interventions taken
        """
        self.step_count += 1
        
        # Extract gradient norms
        total_grad_norm = 0.0
        for param in self.sae.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        # Compute additional metrics
        with torch.no_grad():
            reconstruction, features = self.sae(batch_data)
            
            # Reconstruction quality metrics
            cosine_sim = F.cosine_similarity(batch_data, reconstruction, dim=1).mean().item()
            
            # Explained variance
            total_var = torch.var(batch_data, dim=0).sum().item()
            residual_var = torch.var(batch_data - reconstruction, dim=0).sum().item()
            explained_var = 1.0 - (residual_var / total_var) if total_var > 0 else 0.0
            
            # Sparsity metrics
            sparsity_l0 = (features > 0).float().mean().item()
            
            # Dead neuron fraction
            dead_mask = self.sae.get_dead_neurons()
            dead_fraction = dead_mask.float().mean().item()
        
        # Combine all metrics
        comprehensive_metrics = {
            **{k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()},
            **metrics_dict,
            'cosine_similarity': cosine_sim,
            'explained_variance': explained_var,
            'sparsity_l0': sparsity_l0,
            'dead_fraction': dead_fraction,
            'gradient_norm': total_grad_norm,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'sparsity_coeff': getattr(self.sae, 'current_sparsity_coeff', 0.0)
        }
        
        # Update dashboard
        self.dashboard.update_metrics(step, comprehensive_metrics)
        
        # Automated interventions
        interventions = self._perform_automated_interventions(
            optimizer, batch_data, comprehensive_metrics
        )
        
        # Quality assessment
        quality_report = self.dashboard.get_quality_report()
        
        return {
            'metrics': comprehensive_metrics,
            'quality_report': quality_report,
            'interventions': interventions,
            'should_stop': self._should_stop_training(quality_report)
        }
    
    def _perform_automated_interventions(self, optimizer, batch_data: torch.Tensor, 
                                       metrics: Dict[str, float]) -> List[str]:
        """Perform automated training interventions based on quality metrics"""
        interventions = []
        
        # Dead neuron resampling
        if (self.auto_resample_dead and 
            metrics['dead_fraction'] > self.thresholds.resample_trigger_fraction):
            
            dead_mask = self.sae.get_dead_neurons()
            n_dead = dead_mask.sum().item()
            
            if n_dead > 0:
                self.sae.resample_dead_neurons(batch_data, dead_mask)
                interventions.append(f"Resampled {n_dead} dead neurons")
                
                self.resampling_history.append({
                    'step': self.step_count,
                    'n_resampled': n_dead,
                    'dead_fraction': metrics['dead_fraction']
                })
        
        # Gradient norm intervention
        if (self.auto_reduce_lr and 
            metrics['gradient_norm'] > self.thresholds.gradient_norm_threshold):
            
            old_lr = optimizer.param_groups[0]['lr']
            new_lr = old_lr * 0.5
            optimizer.param_groups[0]['lr'] = new_lr
            
            interventions.append(f"Reduced learning rate: {old_lr:.2e} -> {new_lr:.2e}")
            
            self.intervention_history.append({
                'step': self.step_count,
                'type': 'lr_reduction',
                'old_value': old_lr,
                'new_value': new_lr
            })
        
        return interventions
    
    def _should_stop_training(self, quality_report: Dict[str, Any]) -> bool:
        """Determine if training should be stopped based on quality criteria"""
        # Stop if quality consistently fails
        if quality_report['status'] == 'fail':
            self.quality_violations.append(self.step_count)
            
            # Stop if quality has been failing for too long
            if len(self.quality_violations) >= 10:
                recent_violations = [v for v in self.quality_violations if v > self.step_count - 50]
                if len(recent_violations) >= 8:  # 8 out of last 10 steps failed
                    return True
        else:
            # Reset violations if quality improves
            self.quality_violations = []
        
        return False
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        return {
            'total_steps_monitored': self.step_count,
            'resampling_events': len(self.resampling_history),
            'interventions': len(self.intervention_history),
            'quality_violations': len(self.quality_violations),
            'dashboard_metrics': self.dashboard.get_quality_report(),
            'resampling_history': self.resampling_history,
            'intervention_history': self.intervention_history
        }


class SAEFeatureAnalyzer:
    """
    Comprehensive post-training feature analysis and validation
    
    Provides interpretability scoring, semantic validation, and quality assessment
    """
    
    def __init__(self, sae_model, activation_data: torch.Tensor, 
                 thresholds: Optional[TrainingQualityThresholds] = None):
        self.sae = sae_model
        self.activation_data = activation_data.to(sae_model.cfg.device)
        self.thresholds = thresholds or TrainingQualityThresholds()
        
        # Compute all features once for efficiency
        with torch.no_grad():
            self.sae.eval()
            self.reconstruction, self.features = self.sae(self.activation_data)
        
        self.n_features = self.features.shape[1]
        self.n_samples = self.features.shape[0]
        
        # Feature quality cache
        self._feature_quality_cache = {}
    
    def analyze_feature_quality(self, feature_idx: int, 
                              max_examples: int = 100) -> FeatureQualityMetrics:
        """Comprehensive quality analysis for a single feature"""
        
        if feature_idx in self._feature_quality_cache:
            return self._feature_quality_cache[feature_idx]
        
        feature_activations = self.features[:, feature_idx]
        
        # Basic statistics
        activation_frequency = (feature_activations > 0).float().mean().item()
        mean_activation = feature_activations.mean().item()
        max_activation = feature_activations.max().item()
        
        # Coherence score: consistency of activations across similar inputs
        coherence_score = self._compute_feature_coherence(feature_idx)
        
        # Semantic consistency: stability across related contexts
        semantic_consistency = self._compute_semantic_consistency(feature_idx)
        
        # Redundancy score: similarity to other features
        redundancy_score = self._compute_feature_redundancy(feature_idx)
        
        # Overall interpretability score
        interpretability_score = self._compute_interpretability_score(
            coherence_score, semantic_consistency, redundancy_score, activation_frequency
        )
        
        # Confidence in feature quality
        confidence_score = self._compute_confidence_score(
            feature_activations, coherence_score, semantic_consistency
        )
        
        # Quality assessment
        quality_passed = self._assess_feature_quality(
            activation_frequency, coherence_score, semantic_consistency, 
            redundancy_score, interpretability_score
        )
        
        metrics = FeatureQualityMetrics(
            feature_idx=feature_idx,
            activation_frequency=activation_frequency,
            mean_activation=mean_activation,
            max_activation=max_activation,
            coherence_score=coherence_score,
            semantic_consistency=semantic_consistency,
            redundancy_score=redundancy_score,
            interpretability_score=interpretability_score,
            confidence_score=confidence_score,
            quality_passed=quality_passed
        )
        
        self._feature_quality_cache[feature_idx] = metrics
        return metrics
    
    def _compute_feature_coherence(self, feature_idx: int) -> float:
        """Compute coherence score for feature activations"""
        feature_activations = self.features[:, feature_idx]
        
        # Get top activating examples
        active_mask = feature_activations > 0
        if active_mask.sum() < 10:  # Need minimum examples
            return 0.0
        
        active_activations = self.activation_data[active_mask]
        
        # Compute pairwise similarities among active examples
        if active_activations.shape[0] > 100:  # Sample for efficiency
            indices = torch.randperm(active_activations.shape[0])[:100]
            active_activations = active_activations[indices]
        
        # Cosine similarity matrix
        similarities = F.cosine_similarity(
            active_activations.unsqueeze(1), 
            active_activations.unsqueeze(0), 
            dim=2
        )
        
        # Average similarity (excluding diagonal)
        mask = ~torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)
        coherence = similarities[mask].mean().item()
        
        return max(0.0, coherence)  # Ensure non-negative
    
    def _compute_semantic_consistency(self, feature_idx: int) -> float:
        """Compute semantic consistency across activation contexts"""
        feature_activations = self.features[:, feature_idx]
        
        # Get top and bottom activating examples
        top_k = 50
        if (feature_activations > 0).sum() < top_k:
            return 0.0
        
        top_indices = torch.topk(feature_activations, min(top_k, len(feature_activations))).indices
        top_activations = self.activation_data[top_indices]
        
        # Compute activation consistency across top examples
        activation_std = torch.std(top_activations, dim=0).mean().item()
        activation_mean = torch.mean(top_activations, dim=0).norm().item()
        
        # Consistency is inverse of relative standard deviation
        if activation_mean > 0:
            consistency = max(0.0, 1.0 - (activation_std / activation_mean))
        else:
            consistency = 0.0
        
        return consistency
    
    def _compute_feature_redundancy(self, feature_idx: int) -> float:
        """Compute redundancy with other features"""
        feature_vector = self.features[:, feature_idx]
        
        # Sample other features for efficiency
        other_indices = [i for i in range(self.n_features) if i != feature_idx]
        if len(other_indices) > 500:  # Sample for large SAEs
            other_indices = np.random.choice(other_indices, 500, replace=False)
        
        other_features = self.features[:, other_indices]
        
        # Compute correlations with other features
        correlations = []
        for i in range(other_features.shape[1]):
            other_vector = other_features[:, i]
            if torch.std(feature_vector) > 1e-6 and torch.std(other_vector) > 1e-6:
                corr = torch.corrcoef(torch.stack([feature_vector, other_vector]))[0, 1].item()
                correlations.append(abs(corr))
        
        if not correlations:
            return 0.0
        
        # Return maximum correlation (redundancy)
        return max(correlations)
    
    def _compute_interpretability_score(self, coherence: float, semantic_consistency: float, 
                                      redundancy: float, activation_freq: float) -> float:
        """Combine metrics into interpretability score"""
        # Weight different aspects of interpretability
        weights = {
            'coherence': 0.3,
            'semantic': 0.3,
            'redundancy': 0.2,  # Lower redundancy is better
            'activation': 0.2
        }
        
        # Normalize activation frequency (too high or too low is bad)
        freq_score = 1.0 - abs(activation_freq - self.thresholds.target_sparsity_l0) / self.thresholds.target_sparsity_l0
        freq_score = max(0.0, freq_score)
        
        # Combine scores
        score = (
            weights['coherence'] * coherence +
            weights['semantic'] * semantic_consistency +
            weights['redundancy'] * (1.0 - redundancy) +  # Invert redundancy
            weights['activation'] * freq_score
        )
        
        return max(0.0, min(1.0, score))
    
    def _compute_confidence_score(self, feature_activations: torch.Tensor, 
                                coherence: float, semantic_consistency: float) -> float:
        """Compute confidence in feature quality assessment"""
        # Number of active examples
        n_active = (feature_activations > 0).sum().item()
        activation_confidence = min(1.0, n_active / 100.0)  # Full confidence at 100+ examples
        
        # Stability of measurements
        stability_confidence = (coherence + semantic_consistency) / 2.0
        
        # Overall confidence
        confidence = (activation_confidence + stability_confidence) / 2.0
        
        return confidence
    
    def _assess_feature_quality(self, activation_freq: float, coherence: float, 
                              semantic_consistency: float, redundancy: float, 
                              interpretability: float) -> bool:
        """Determine if feature passes quality thresholds"""
        checks = [
            activation_freq >= self.thresholds.min_sparsity_l0,
            activation_freq <= self.thresholds.max_sparsity_l0,
            coherence >= self.thresholds.min_feature_coherence,
            semantic_consistency >= self.thresholds.min_semantic_consistency,
            redundancy <= self.thresholds.max_feature_redundancy
        ]
        
        return all(checks)
    
    def extract_max_activating_examples(self, feature_idx: int, 
                                      n_examples: int = 20) -> Dict[str, Any]:
        """Extract and analyze max activating examples for a feature"""
        feature_activations = self.features[:, feature_idx]
        
        # Get top activating examples
        top_indices = torch.topk(feature_activations, min(n_examples, len(feature_activations))).indices
        top_activations = feature_activations[top_indices]
        top_examples = self.activation_data[top_indices]
        
        # Confidence scoring for examples
        confidence_scores = []
        for i, idx in enumerate(top_indices):
            # Confidence based on activation strength and context similarity
            activation_strength = top_activations[i].item()
            
            # Compare to nearby examples for context stability
            context_stability = self._compute_context_stability(idx.item(), feature_idx)
            
            confidence = (activation_strength / feature_activations.max().item()) * context_stability
            confidence_scores.append(confidence)
        
        return {
            'feature_idx': feature_idx,
            'top_indices': top_indices.cpu().numpy().tolist(),
            'top_activations': top_activations.cpu().numpy().tolist(),
            'confidence_scores': confidence_scores,
            'top_examples': top_examples.cpu().numpy(),
            'mean_activation': feature_activations.mean().item(),
            'activation_percentile_95': torch.quantile(feature_activations, 0.95).item()
        }
    
    def _compute_context_stability(self, example_idx: int, feature_idx: int) -> float:
        """Compute stability of feature activation in example context"""
        # Get nearby examples (simple approach - could be improved with proper similarity)
        window_size = 10
        start_idx = max(0, example_idx - window_size)
        end_idx = min(self.n_samples, example_idx + window_size + 1)
        
        nearby_activations = self.features[start_idx:end_idx, feature_idx]
        target_activation = self.features[example_idx, feature_idx]
        
        if len(nearby_activations) < 3:
            return 1.0
        
        # Stability is inverse of relative variation
        std_dev = torch.std(nearby_activations).item()
        mean_val = torch.mean(nearby_activations).item()
        
        if mean_val > 0:
            stability = max(0.0, 1.0 - (std_dev / mean_val))
        else:
            stability = 0.0
        
        return stability
    
    def generate_feature_heatmaps(self, feature_indices: List[int], 
                                save_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Generate activation heatmaps for specified features"""
        heatmaps = {}
        
        for feature_idx in feature_indices:
            feature_activations = self.features[:, feature_idx].cpu().numpy()
            
            # Reshape for heatmap (assuming sequential structure)
            # This is a simplified approach - real implementation would depend on data structure
            heatmap_size = int(np.sqrt(min(len(feature_activations), 10000)))
            if len(feature_activations) >= heatmap_size ** 2:
                heatmap = feature_activations[:heatmap_size**2].reshape(heatmap_size, heatmap_size)
            else:
                # Pad with zeros if needed
                padded = np.pad(feature_activations, 
                              (0, max(0, heatmap_size**2 - len(feature_activations))))
                heatmap = padded[:heatmap_size**2].reshape(heatmap_size, heatmap_size)
            
            heatmaps[f'feature_{feature_idx}'] = heatmap
            
            # Save heatmap visualization
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(heatmap, cmap='viridis', cbar=True)
                plt.title(f'Feature {feature_idx} Activation Heatmap')
                plt.savefig(os.path.join(save_dir, f'feature_{feature_idx}_heatmap.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        return heatmaps
    
    def analyze_feature_diversity(self, sample_size: Optional[int] = 1000) -> Dict[str, Any]:
        """Analyze diversity and clustering patterns among features"""
        # Sample features if SAE is very large
        if self.n_features > sample_size and sample_size:
            feature_indices = np.random.choice(self.n_features, sample_size, replace=False)
            sample_features = self.features[:, feature_indices]
        else:
            feature_indices = np.arange(self.n_features)
            sample_features = self.features
        
        # Compute feature correlation matrix
        feature_correlations = torch.corrcoef(sample_features.T).cpu().numpy()
        
        # Remove NaN values
        nan_mask = ~np.isnan(feature_correlations)
        feature_correlations = np.where(nan_mask, feature_correlations, 0)
        
        # Clustering analysis
        n_clusters_range = range(2, min(21, len(feature_indices) // 10))
        if len(n_clusters_range) == 0:
            n_clusters_range = [2]
        
        best_n_clusters = 2
        best_silhouette = -1
        
        clustering_results = {}
        
        for n_clusters in n_clusters_range:
            if n_clusters >= len(feature_indices):
                continue
                
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(sample_features.T.cpu().numpy())
                
                silhouette_avg = silhouette_score(sample_features.T.cpu().numpy(), cluster_labels)
                clustering_results[n_clusters] = {
                    'labels': cluster_labels,
                    'silhouette_score': silhouette_avg,
                    'cluster_centers': kmeans.cluster_centers_
                }
                
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_n_clusters = n_clusters
                    
            except Exception as e:
                continue
        
        # Redundancy analysis
        correlation_values = feature_correlations[np.triu_indices_from(feature_correlations, k=1)]
        high_correlation_pairs = np.sum(np.abs(correlation_values) > self.thresholds.max_feature_redundancy)
        
        return {
            'correlation_matrix': feature_correlations,
            'correlation_stats': {
                'mean_correlation': np.mean(np.abs(correlation_values)),
                'max_correlation': np.max(np.abs(correlation_values)),
                'high_correlation_pairs': int(high_correlation_pairs),
                'redundancy_fraction': high_correlation_pairs / len(correlation_values)
            },
            'clustering': {
                'best_n_clusters': best_n_clusters,
                'best_silhouette_score': best_silhouette,
                'all_results': clustering_results
            },
            'diversity_score': 1.0 - np.mean(np.abs(correlation_values))  # Higher is more diverse
        }
    
    def detect_semantic_patterns(self, min_pattern_size: int = 5) -> Dict[str, Any]:
        """Detect semantic patterns across learned features"""
        patterns = {}
        
        # Group features by activation patterns
        activation_patterns = {}
        
        for i in range(self.n_features):
            feature_activations = self.features[:, i]
            
            # Create activation signature (simplified)
            active_samples = (feature_activations > 0).nonzero(as_tuple=True)[0]
            
            if len(active_samples) >= min_pattern_size:
                # Create a signature based on which samples activate
                signature = tuple(sorted(active_samples[:min_pattern_size].cpu().numpy().tolist()))
                
                if signature not in activation_patterns:
                    activation_patterns[signature] = []
                activation_patterns[signature].append(i)
        
        # Find significant patterns
        significant_patterns = {
            sig: features for sig, features in activation_patterns.items() 
            if len(features) >= min_pattern_size
        }
        
        # Analyze pattern coherence
        pattern_analysis = {}
        for pattern_id, feature_list in significant_patterns.items():
            if len(feature_list) < 2:
                continue
            
            # Compute inter-feature correlations within pattern
            pattern_features = self.features[:, feature_list]
            pattern_correlations = torch.corrcoef(pattern_features.T).cpu().numpy()
            
            # Remove diagonal and compute statistics
            off_diagonal = pattern_correlations[np.triu_indices_from(pattern_correlations, k=1)]
            
            pattern_analysis[f'pattern_{len(pattern_analysis)}'] = {
                'features': feature_list,
                'n_features': len(feature_list),
                'mean_correlation': np.mean(off_diagonal),
                'coherence_score': np.mean(off_diagonal > 0.3),  # Fraction with good correlation
                'pattern_signature': pattern_id
            }
        
        return {
            'n_patterns_detected': len(significant_patterns),
            'pattern_analysis': pattern_analysis,
            'pattern_coverage': sum(len(features) for features in significant_patterns.values()) / self.n_features
        }
    
    def generate_comprehensive_report(self, save_dir: Optional[str] = None, 
                                    n_top_features: int = 50) -> Dict[str, Any]:
        """Generate comprehensive feature validation report"""
        
        print("Generating comprehensive SAE feature analysis report...")
        
        # Analyze top features by activation frequency
        activation_frequencies = (self.features > 0).float().mean(dim=0)
        top_feature_indices = torch.topk(activation_frequencies, min(n_top_features, self.n_features)).indices
        
        # Individual feature analysis
        print(f"Analyzing quality of top {len(top_feature_indices)} features...")
        feature_analyses = {}
        quality_summary = {'passed': 0, 'failed': 0}
        
        for i, feature_idx in enumerate(tqdm(top_feature_indices, desc="Analyzing features")):
            feature_idx = feature_idx.item()
            quality_metrics = self.analyze_feature_quality(feature_idx)
            
            feature_analyses[f'feature_{feature_idx}'] = asdict(quality_metrics)
            
            if quality_metrics.quality_passed:
                quality_summary['passed'] += 1
            else:
                quality_summary['failed'] += 1
        
        # Diversity analysis
        print("Analyzing feature diversity and clustering...")
        diversity_analysis = self.analyze_feature_diversity()
        
        # Semantic pattern detection
        print("Detecting semantic patterns...")
        semantic_patterns = self.detect_semantic_patterns()
        
        # Overall statistics
        overall_stats = {
            'n_features_analyzed': len(top_feature_indices),
            'quality_pass_rate': quality_summary['passed'] / len(top_feature_indices),
            'mean_activation_frequency': activation_frequencies.mean().item(),
            'sparsity_compliance': torch.sum(
                (activation_frequencies >= self.thresholds.min_sparsity_l0) & 
                (activation_frequencies <= self.thresholds.max_sparsity_l0)
            ).item() / self.n_features,
            'dead_feature_fraction': torch.sum(
                activation_frequencies <= self.thresholds.dead_neuron_threshold
            ).item() / self.n_features
        }
        
        # Generate pass/fail assessment
        overall_pass_criteria = {
            'quality_pass_rate': overall_stats['quality_pass_rate'] >= 0.7,  # 70% features should pass
            'sparsity_compliance': overall_stats['sparsity_compliance'] >= 0.8,  # 80% within range
            'dead_feature_fraction': overall_stats['dead_feature_fraction'] <= 0.2,  # <20% dead
            'diversity_score': diversity_analysis['diversity_score'] >= 0.3,  # Reasonable diversity
            'redundancy_check': diversity_analysis['correlation_stats']['redundancy_fraction'] <= 0.3
        }
        
        overall_assessment = {
            'overall_pass': all(overall_pass_criteria.values()),
            'criteria_results': overall_pass_criteria,
            'recommendations': self._generate_feature_recommendations(
                overall_stats, diversity_analysis, overall_pass_criteria
            )
        }
        
        # Compile complete report
        complete_report = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'n_samples': self.n_samples,
                'n_features_total': self.n_features,
                'n_features_analyzed': len(top_feature_indices)
            },
            'overall_statistics': overall_stats,
            'overall_assessment': overall_assessment,
            'feature_analyses': feature_analyses,
            'diversity_analysis': diversity_analysis,
            'semantic_patterns': semantic_patterns,
            'quality_thresholds': asdict(self.thresholds)
        }
        
        # Save report
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save JSON report
            report_path = os.path.join(save_dir, 'sae_feature_analysis_report.json')
            with open(report_path, 'w') as f:
                json.dump(complete_report, f, indent=2, default=str)
            
            # Generate visualizations
            self._generate_analysis_visualizations(complete_report, save_dir)
            
            print(f"Complete analysis report saved to {save_dir}")
        
        return complete_report
    
    def _generate_feature_recommendations(self, stats: Dict, diversity: Dict, 
                                        criteria: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if not criteria['quality_pass_rate']:
            recommendations.append(
                f"Feature quality is low ({stats['quality_pass_rate']:.1%}). "
                "Consider increasing training epochs or adjusting sparsity coefficient."
            )
        
        if not criteria['sparsity_compliance']:
            recommendations.append(
                f"Sparsity compliance is poor ({stats['sparsity_compliance']:.1%}). "
                "Adjust sparsity coefficient to target 5-15% activation range."
            )
        
        if not criteria['dead_feature_fraction']:
            recommendations.append(
                f"Too many dead features ({stats['dead_feature_fraction']:.1%}). "
                "Implement more aggressive dead neuron resampling."
            )
        
        if not criteria['diversity_score']:
            recommendations.append(
                f"Low feature diversity ({diversity['diversity_score']:.2f}). "
                "Consider increasing SAE width or reducing regularization."
            )
        
        if not criteria['redundancy_check']:
            recommendations.append(
                f"High feature redundancy ({diversity['correlation_stats']['redundancy_fraction']:.1%}). "
                "Consider adding orthogonality constraints or increasing sparsity."
            )
        
        return recommendations
    
    def _generate_analysis_visualizations(self, report: Dict, save_dir: str):
        """Generate visualization plots for the analysis report"""
        try:
            # Feature quality distribution
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('SAE Feature Analysis Summary', fontsize=16)
            
            # Extract metrics from feature analyses
            feature_metrics = list(report['feature_analyses'].values())
            
            # Quality scores
            interpretability_scores = [f['interpretability_score'] for f in feature_metrics]
            coherence_scores = [f['coherence_score'] for f in feature_metrics]
            activation_frequencies = [f['activation_frequency'] for f in feature_metrics]
            
            # Plot 1: Interpretability scores
            axes[0, 0].hist(interpretability_scores, bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(x=0.5, color='red', linestyle='--', label='Good threshold')
            axes[0, 0].set_xlabel('Interpretability Score')
            axes[0, 0].set_ylabel('Number of Features')
            axes[0, 0].set_title('Feature Interpretability Distribution')
            axes[0, 0].legend()
            
            # Plot 2: Coherence scores
            axes[0, 1].hist(coherence_scores, bins=20, alpha=0.7, edgecolor='black', color='green')
            axes[0, 1].axvline(x=self.thresholds.min_feature_coherence, color='red', linestyle='--', label='Threshold')
            axes[0, 1].set_xlabel('Coherence Score')
            axes[0, 1].set_ylabel('Number of Features')
            axes[0, 1].set_title('Feature Coherence Distribution')
            axes[0, 1].legend()
            
            # Plot 3: Activation frequencies
            axes[0, 2].hist(activation_frequencies, bins=20, alpha=0.7, edgecolor='black', color='orange')
            axes[0, 2].axvline(x=self.thresholds.min_sparsity_l0, color='red', linestyle='--', label='Min')
            axes[0, 2].axvline(x=self.thresholds.max_sparsity_l0, color='red', linestyle='--', label='Max')
            axes[0, 2].axvline(x=self.thresholds.target_sparsity_l0, color='green', linestyle='-', label='Target')
            axes[0, 2].set_xlabel('Activation Frequency')
            axes[0, 2].set_ylabel('Number of Features')
            axes[0, 2].set_title('Feature Activation Distribution')
            axes[0, 2].legend()
            
            # Plot 4: Feature correlation heatmap
            corr_matrix = report['diversity_analysis']['correlation_matrix']
            if corr_matrix.shape[0] <= 100:  # Only plot if manageable size
                im = axes[1, 0].imshow(np.abs(corr_matrix), cmap='viridis', aspect='auto')
                axes[1, 0].set_title('Feature Correlation Matrix')
                plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
            else:
                axes[1, 0].text(0.5, 0.5, f'Correlation matrix\n({corr_matrix.shape[0]} x {corr_matrix.shape[1]})\ntoo large to display', 
                              ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Feature Correlation Matrix')
            
            # Plot 5: Quality pass/fail pie chart
            quality_data = [report['overall_assessment']['criteria_results'][k] for k in report['overall_assessment']['criteria_results']]
            quality_labels = list(report['overall_assessment']['criteria_results'].keys())
            passed = sum(quality_data)
            failed = len(quality_data) - passed
            
            axes[1, 1].pie([passed, failed], labels=['Passed', 'Failed'], colors=['green', 'red'], autopct='%1.1f%%')
            axes[1, 1].set_title('Quality Criteria Results')
            
            # Plot 6: Overall statistics
            stats = report['overall_statistics']
            stat_names = ['Quality Pass Rate', 'Sparsity Compliance', 'Dead Features', 'Diversity Score']
            stat_values = [
                stats['quality_pass_rate'],
                stats['sparsity_compliance'], 
                1 - stats['dead_feature_fraction'],  # Invert for positive visualization
                report['diversity_analysis']['diversity_score']
            ]
            
            bars = axes[1, 2].bar(range(len(stat_names)), stat_values, color=['blue', 'green', 'orange', 'purple'], alpha=0.7)
            axes[1, 2].set_xticks(range(len(stat_names)))
            axes[1, 2].set_xticklabels(stat_names, rotation=45, ha='right')
            axes[1, 2].set_ylabel('Score')
            axes[1, 2].set_title('Overall Performance Metrics')
            axes[1, 2].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, stat_values):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'feature_analysis_summary.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Analysis visualizations saved to {save_dir}")
            
        except Exception as e:
            print(f"Could not generate visualizations: {e}")


# Integration class for complete monitoring pipeline
class ComprehensiveSAEMonitor:
    """
    Complete SAE training and analysis monitoring system
    
    Integrates real-time training monitoring with post-training feature validation
    """
    
    def __init__(self, sae_model, activation_data: torch.Tensor, 
                 thresholds: Optional[TrainingQualityThresholds] = None):
        self.sae = sae_model
        self.activation_data = activation_data
        self.thresholds = thresholds or TrainingQualityThresholds()
        
        # Initialize monitoring components
        self.training_monitor = SAETrainingMonitor(sae_model, self.thresholds)
        self.feature_analyzer = None  # Initialize after training
        
        # Results storage
        self.monitoring_results = {}
        self.analysis_results = {}
    
    def monitor_training(self, optimizer, dataloader, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Monitor complete training process with real-time quality assurance
        """
        print("Starting comprehensive SAE training monitoring...")
        
        self.sae.train()
        step = 0
        should_stop = False
        
        for epoch in range(100):  # Max epochs
            for batch_idx, (batch_data,) in enumerate(dataloader):
                if max_steps and step >= max_steps:
                    should_stop = True
                    break
                
                batch_data = batch_data.to(self.sae.cfg.device)
                
                # Forward pass
                reconstruction, features = self.sae(batch_data)
                
                # Compute losses
                loss_dict = self.sae.compute_loss(
                    batch_data, reconstruction, features, 
                    getattr(self.sae, 'current_sparsity_coeff', 0.001)
                )
                
                # Compute additional metrics
                sparsity_metrics = self.sae.get_sparsity_metrics(features)
                reconstruction_metrics = self.sae.get_reconstruction_metrics(batch_data, reconstruction)
                
                # Monitor step
                monitoring_result = self.training_monitor.monitor_training_step(
                    step, optimizer, batch_data, loss_dict, 
                    {**sparsity_metrics, **reconstruction_metrics}
                )
                
                # Perform backward pass
                optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.sae.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Check for stopping conditions
                if monitoring_result['should_stop']:
                    print(f"Training stopped early at step {step} due to quality issues")
                    should_stop = True
                    break
                
                step += 1
            
            if should_stop:
                break
        
        # Store monitoring results
        self.monitoring_results = self.training_monitor.get_monitoring_summary()
        
        print(f"Training monitoring completed after {step} steps")
        return self.monitoring_results
    
    def analyze_features(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive post-training feature analysis
        """
        print("Starting post-training feature analysis...")
        
        # Initialize feature analyzer with current activation data
        self.feature_analyzer = SAEFeatureAnalyzer(self.sae, self.activation_data, self.thresholds)
        
        # Generate comprehensive analysis report
        self.analysis_results = self.feature_analyzer.generate_comprehensive_report(save_dir)
        
        print("Feature analysis completed")
        return self.analysis_results
    
    def get_complete_report(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate complete monitoring and analysis report
        """
        complete_report = {
            'monitoring_summary': self.monitoring_results,
            'feature_analysis': self.analysis_results,
            'overall_assessment': self._generate_overall_assessment(),
            'recommendations': self._generate_comprehensive_recommendations()
        }
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            report_path = os.path.join(save_dir, 'complete_sae_report.json')
            with open(report_path, 'w') as f:
                json.dump(complete_report, f, indent=2, default=str)
            
            print(f"Complete SAE report saved to {report_path}")
        
        return complete_report
    
    def _generate_overall_assessment(self) -> Dict[str, Any]:
        """Generate overall pass/fail assessment"""
        training_quality = self.monitoring_results.get('dashboard_metrics', {}).get('status', 'unknown') == 'pass'
        
        feature_quality = False
        if self.analysis_results:
            feature_quality = self.analysis_results.get('overall_assessment', {}).get('overall_pass', False)
        
        return {
            'training_quality_pass': training_quality,
            'feature_quality_pass': feature_quality,
            'overall_pass': training_quality and feature_quality,
            'ready_for_alignment_analysis': training_quality and feature_quality
        }
    
    def _generate_comprehensive_recommendations(self) -> List[str]:
        """Generate comprehensive recommendations for SAE improvement"""
        recommendations = []
        
        # Training recommendations
        if self.monitoring_results:
            training_recs = self.monitoring_results.get('dashboard_metrics', {}).get('recommendations', [])
            recommendations.extend([f"Training: {rec}" for rec in training_recs])
        
        # Feature recommendations  
        if self.analysis_results:
            feature_recs = self.analysis_results.get('overall_assessment', {}).get('recommendations', [])
            recommendations.extend([f"Features: {rec}" for rec in feature_recs])
        
        # Overall recommendations
        overall_assessment = self._generate_overall_assessment()
        if not overall_assessment['overall_pass']:
            if not overall_assessment['training_quality_pass']:
                recommendations.append("Overall: Improve training stability before proceeding to alignment analysis")
            if not overall_assessment['feature_quality_pass']:
                recommendations.append("Overall: Improve feature interpretability before alignment correlation study")
        else:
            recommendations.append("Overall: SAE is ready for alignment feature correlation analysis")
        
        return recommendations


if __name__ == "__main__":
    # Example usage
    print("SAE Training Monitoring and Feature Validation System")
    print("This module provides comprehensive analysis tools for SAE training quality assurance.")
    print("Import and use the classes in your training pipeline for full monitoring capabilities.")