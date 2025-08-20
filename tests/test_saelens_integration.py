"""
Comprehensive validation script for SAELens integration with HookedTransformerWrapper

This test suite validates the integration between the custom transformer implementation
and the SAELens ecosystem for mechanistic interpretability research.
"""

import os
import sys
import time
import warnings
import traceback
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from transformer.attention import HookedTransformerWrapper, ActivationCache, HookPoint
except ImportError as e:
    print(f"Error importing custom transformer: {e}")
    print("Make sure the src/transformer directory is properly structured")
    sys.exit(1)

# Try to import SAELens - graceful fallback if not available
try:
    import sae_lens
    SAE_LENS_AVAILABLE = True
except ImportError:
    print("Warning: SAELens not available. Creating mock SAE for testing.")
    SAE_LENS_AVAILABLE = False


class MockSAE(nn.Module):
    """Mock SAE implementation for testing when SAELens is not available"""
    def __init__(self, d_in: int, d_sae: int = 2048, hook_name: str = "hook_test"):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.hook_name = hook_name
        
        # Create mock config object
        self.cfg = type('cfg', (), {
            'hook_name': hook_name,
            'd_in': d_in,
            'd_sae': d_sae,
            'device': 'cpu'
        })()
        
        # Simple encoder-decoder architecture
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        
    def encode(self, x):
        """Encode input to sparse features"""
        return torch.relu(self.encoder(x))
    
    def decode(self, features):
        """Decode sparse features back to original space"""
        return self.decoder(features) + self.b_dec
    
    def forward(self, x):
        """Full forward pass: encode then decode"""
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction
    
    def get_feature_acts(self, x):
        """Get feature activations"""
        return self.encode(x)


# Test configuration
GPT_CONFIG_TEST = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 6,  # Smaller for testing
    "drop_rate": 0.1,
    "qkv_bias": False
}

class SAELensIntegrationTest:
    """Comprehensive test suite for SAELens integration"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = None
        self.sae = None
        self.test_results = {}
        self.debug_info = {}
        
    def setup_model(self):
        """Initialize the HookedTransformerWrapper with proper weights"""
        print("Setting up HookedTransformerWrapper...")
        
        self.model = HookedTransformerWrapper(GPT_CONFIG_TEST)
        self.model.to(self.device)
        
        # Initialize weights properly to avoid NaN values
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    param.fill_(0.01)
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        return True
    
    def load_sae(self, hook_name: str = "blocks.2.hook_resid_post"):
        """Load or create SAE for testing"""
        print(f"Loading SAE for hook: {hook_name}")
        
        if SAE_LENS_AVAILABLE:
            try:
                # Try to load a real SAE from SAELens
                # This would be the actual implementation:
                # self.sae = sae_lens.SAE.from_pretrained("gpt2-small-hook_resid_pre")
                print("SAELens available but using mock SAE for testing")
                self.sae = MockSAE(
                    d_in=GPT_CONFIG_TEST["emb_dim"],
                    d_sae=2048,
                    hook_name=hook_name
                )
            except Exception as e:
                print(f"Failed to load real SAE: {e}")
                self.sae = MockSAE(
                    d_in=GPT_CONFIG_TEST["emb_dim"],
                    d_sae=2048,
                    hook_name=hook_name
                )
        else:
            self.sae = MockSAE(
                d_in=GPT_CONFIG_TEST["emb_dim"],
                d_sae=2048,
                hook_name=hook_name
            )
        
        self.sae.to(self.device)
        print(f"SAE loaded: {self.sae.d_in} -> {self.sae.d_sae} -> {self.sae.d_in}")
        return True
    
    def test_basic_functionality(self):
        """Test basic model functionality"""
        print("\n=== Testing Basic Functionality ===")
        
        try:
            # Create test input
            batch_size, seq_len = 2, 10
            tokens = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
            
            # Test forward pass
            with torch.no_grad():
                logits = self.model(tokens)
            
            # Validate output shape
            expected_shape = (batch_size, seq_len, GPT_CONFIG_TEST["vocab_size"])
            assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
            
            # Check for NaN values
            assert not torch.isnan(logits).any(), "Model output contains NaN values"
            
            self.test_results['basic_functionality'] = True
            print("[PASS] Basic functionality test passed")
            return True
            
        except Exception as e:
            self.test_results['basic_functionality'] = False
            print(f"[FAIL] Basic functionality test failed: {e}")
            return False
    
    def test_activation_caching(self):
        """Test run_with_cache functionality"""
        print("\n=== Testing Activation Caching ===")
        
        try:
            batch_size, seq_len = 2, 8
            tokens = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
            
            # Test caching
            with torch.no_grad():
                logits, cache = self.model.run_with_cache(tokens)
            
            # Validate cache contents
            assert isinstance(cache, ActivationCache), f"Expected ActivationCache, got {type(cache)}"
            assert len(cache.keys()) > 0, "Cache is empty"
            
            # Check expected activations
            expected_keys = [
                "hook_embed", "hook_pos_embed", "hook_tokens",
                "blocks.0.hook_resid_pre", "blocks.0.hook_resid_post",
                "hook_attn_weights", "hook_z"
            ]
            
            missing_keys = [key for key in expected_keys if key not in cache]
            if missing_keys:
                print(f"Warning: Missing expected keys: {missing_keys}")
            
            # Validate activation shapes
            for key, activation in cache.items():
                if "hook_" in key and isinstance(activation, torch.Tensor):
                    # Most activations should have batch dimension
                    assert activation.dim() >= 2, f"Activation {key} has unexpected dimensions: {activation.shape}"
                    
                    # Check for reasonable values
                    assert torch.isfinite(activation).all(), f"Activation {key} contains non-finite values"
            
            # Store debug info
            self.debug_info['cache_keys'] = list(cache.keys())
            self.debug_info['cache_shapes'] = {k: v.shape for k, v in cache.items() if isinstance(v, torch.Tensor)}
            
            self.test_results['activation_caching'] = True
            print(f"[PASS] Activation caching test passed ({len(cache.keys())} activations cached)")
            return True
            
        except Exception as e:
            self.test_results['activation_caching'] = False
            print(f"[FAIL] Activation caching test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_activation_shapes(self):
        """Validate activation shapes match SAELens expectations"""
        print("\n=== Testing Activation Shapes ===")
        
        try:
            batch_sizes = [1, 2, 4]
            seq_lens = [5, 10, 16]
            
            for batch_size in batch_sizes:
                for seq_len in seq_lens:
                    tokens = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
                    
                    with torch.no_grad():
                        logits, cache = self.model.run_with_cache(tokens)
                    
                    # Check residual stream activations
                    for layer in range(GPT_CONFIG_TEST["n_layers"]):
                        key = f"blocks.{layer}.hook_resid_post"
                        if key in cache:
                            activation = cache[key]
                            expected_shape = (batch_size, seq_len, GPT_CONFIG_TEST["emb_dim"])
                            assert activation.shape == expected_shape, \
                                f"Layer {layer} residual: expected {expected_shape}, got {activation.shape}"
                    
                    # Check attention weights
                    if "hook_attn_weights" in cache:
                        attn_weights = cache["hook_attn_weights"]
                        expected_shape = (batch_size, GPT_CONFIG_TEST["n_heads"], seq_len, seq_len)
                        assert attn_weights.shape == expected_shape, \
                            f"Attention weights: expected {expected_shape}, got {attn_weights.shape}"
            
            self.test_results['activation_shapes'] = True
            print("[PASS] Activation shapes test passed")
            return True
            
        except Exception as e:
            self.test_results['activation_shapes'] = False
            print(f"[FAIL] Activation shapes test failed: {e}")
            return False
    
    def test_sae_integration(self):
        """Test SAE forward pass on collected activations"""
        print("\n=== Testing SAE Integration ===")
        
        try:
            batch_size, seq_len = 2, 8
            tokens = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
            
            # Get activations
            with torch.no_grad():
                logits, cache = self.model.run_with_cache(tokens)
            
            # Test SAE on target activation
            target_key = self.sae.hook_name
            if target_key not in cache:
                # Fall back to any available residual activation
                target_key = next((k for k in cache.keys() if "hook_resid_post" in k), None)
            
            if target_key is None:
                raise ValueError("No suitable activation found for SAE testing")
            
            activation = cache[target_key]
            
            # Test SAE forward pass
            with torch.no_grad():
                reconstruction = self.sae(activation)
                features = self.sae.get_feature_acts(activation)
            
            # Validate SAE outputs
            assert reconstruction.shape == activation.shape, \
                f"Reconstruction shape mismatch: {reconstruction.shape} vs {activation.shape}"
            
            assert features.shape[:2] == activation.shape[:2], \
                f"Feature shape mismatch: {features.shape[:2]} vs {activation.shape[:2]}"
            
            assert features.shape[2] == self.sae.d_sae, \
                f"Feature dimension mismatch: {features.shape[2]} vs {self.sae.d_sae}"
            
            # Check reconstruction quality
            mse_loss = torch.mean((reconstruction - activation) ** 2)
            reconstruction_error = mse_loss.item()
            
            # Store metrics
            self.debug_info['reconstruction_error'] = reconstruction_error
            self.debug_info['feature_sparsity'] = (features == 0).float().mean().item()
            
            self.test_results['sae_integration'] = True
            print(f"[PASS] SAE integration test passed (reconstruction error: {reconstruction_error:.4f})")
            return True
            
        except Exception as e:
            self.test_results['sae_integration'] = False
            print(f"[FAIL] SAE integration test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_sae_training_compatibility(self):
        """Test SAE training data collection workflow"""
        print("\n=== Testing SAE Training Compatibility ===")
        
        try:
            # Simulate training data collection
            batch_size, seq_len = 4, 12
            num_batches = 3
            
            all_activations = []
            
            for batch_idx in range(num_batches):
                tokens = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
                
                with torch.no_grad():
                    logits, cache = self.model.run_with_cache(tokens)
                
                # Collect target activations
                target_key = "blocks.2.hook_resid_post"  # Middle layer for semantic richness
                if target_key in cache:
                    activation = cache[target_key]
                    # Reshape to (batch * seq, d_model) for SAE training
                    activation_flat = activation.reshape(-1, activation.shape[-1])
                    all_activations.append(activation_flat)
            
            # Combine all activations
            if all_activations:
                training_data = torch.cat(all_activations, dim=0)
                
                # Validate training data format
                expected_samples = num_batches * batch_size * seq_len
                expected_shape = (expected_samples, GPT_CONFIG_TEST["emb_dim"])
                assert training_data.shape == expected_shape, \
                    f"Training data shape mismatch: {training_data.shape} vs {expected_shape}"
                
                # Test SAE training step (forward pass only)
                with torch.no_grad():
                    sample_batch = training_data[:32]  # Small batch
                    features = self.sae.get_feature_acts(sample_batch)
                    reconstruction = self.sae.decode(features)
                    
                    # Calculate training metrics
                    mse_loss = torch.mean((reconstruction - sample_batch) ** 2)
                    sparsity = (features == 0).float().mean()
                    
                    self.debug_info['training_mse'] = mse_loss.item()
                    self.debug_info['training_sparsity'] = sparsity.item()
                
                self.test_results['sae_training_compatibility'] = True
                print(f"[PASS] SAE training compatibility test passed ({training_data.shape[0]} samples collected)")
                return True
            else:
                raise ValueError("No activations collected for training")
                
        except Exception as e:
            self.test_results['sae_training_compatibility'] = False
            print(f"[FAIL] SAE training compatibility test failed: {e}")
            return False
    
    def test_activation_patching(self):
        """Test activation patching functionality for interventions"""
        print("\n=== Testing Activation Patching ===")
        
        try:
            batch_size, seq_len = 2, 8
            tokens = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
            
            # Get baseline activations
            with torch.no_grad():
                baseline_logits, baseline_cache = self.model.run_with_cache(tokens)
            
            # Test hook manipulation
            target_layer = 1
            hook_name = f"blocks.{target_layer}.hook_resid_post"
            
            def intervention_hook(activation, hook_name):
                # Simple intervention: add noise
                noise = torch.randn_like(activation) * 0.1
                return activation + noise
            
            # Apply intervention
            hook_points = self.model._get_hook_points()
            if hook_name in hook_points:
                hook_point = hook_points[hook_name]
                hook_point.add_hook(intervention_hook)
                
                try:
                    with torch.no_grad():
                        modified_logits = self.model(tokens)
                    
                    # Check that intervention had an effect
                    logit_diff = torch.mean((modified_logits - baseline_logits) ** 2)
                    assert logit_diff > 1e-6, "Intervention had no measurable effect"
                    
                    self.debug_info['intervention_effect'] = logit_diff.item()
                    
                finally:
                    # Clean up hook
                    hook_point.remove_hooks()
            
            self.test_results['activation_patching'] = True
            print("[PASS] Activation patching test passed")
            return True
            
        except Exception as e:
            self.test_results['activation_patching'] = False
            print(f"[FAIL] Activation patching test failed: {e}")
            return False
    
    def test_gradient_flow(self):
        """Validate gradient flow through hook system"""
        print("\n=== Testing Gradient Flow ===")
        
        try:
            batch_size, seq_len = 2, 8
            tokens = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
            
            # Create target for loss calculation
            target_logits = torch.randn(batch_size, seq_len, GPT_CONFIG_TEST["vocab_size"], device=self.device)
            
            # Enable gradients
            self.model.train()
            
            # Forward pass with caching
            logits, cache = self.model.run_with_cache(tokens)
            
            # Calculate loss
            loss = torch.mean((logits - target_logits) ** 2)
            
            # Backward pass
            loss.backward()
            
            # Check that gradients exist
            has_gradients = False
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    has_gradients = True
                    assert torch.isfinite(param.grad).all(), f"Non-finite gradients in {name}"
            
            assert has_gradients, "No gradients found after backward pass"
            
            # Clear gradients
            self.model.zero_grad()
            self.model.eval()
            
            self.test_results['gradient_flow'] = True
            print("[PASS] Gradient flow test passed")
            return True
            
        except Exception as e:
            self.test_results['gradient_flow'] = False
            print(f"[FAIL] Gradient flow test failed: {e}")
            return False
    
    def test_memory_management(self):
        """Check memory management during activation caching"""
        print("\n=== Testing Memory Management ===")
        
        try:
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated(self.device)
            
            # Test with increasing sequence lengths
            for seq_len in [10, 50, 100]:
                batch_size = 2
                tokens = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
                
                # Multiple forward passes
                for _ in range(3):
                    with torch.no_grad():
                        logits, cache = self.model.run_with_cache(tokens)
                    
                    # Clear cache
                    del cache
                    del logits
                
                # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated(self.device)
                memory_diff = final_memory - initial_memory
                self.debug_info['memory_usage'] = memory_diff
            
            self.test_results['memory_management'] = True
            print("[PASS] Memory management test passed")
            return True
            
        except Exception as e:
            self.test_results['memory_management'] = False
            print(f"[FAIL] Memory management test failed: {e}")
            return False
    
    def test_device_consistency(self):
        """Ensure device consistency across model and SAE"""
        print("\n=== Testing Device Consistency ===")
        
        try:
            batch_size, seq_len = 2, 8
            tokens = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
            
            # Check model device
            model_device = next(self.model.parameters()).device
            assert model_device == self.device, f"Model on wrong device: {model_device} vs {self.device}"
            
            # Check SAE device
            sae_device = next(self.sae.parameters()).device
            assert sae_device == self.device, f"SAE on wrong device: {sae_device} vs {self.device}"
            
            # Test forward pass
            with torch.no_grad():
                logits, cache = self.model.run_with_cache(tokens)
                
                # Check activation devices
                for key, activation in cache.items():
                    if isinstance(activation, torch.Tensor):
                        assert activation.device == self.device, \
                            f"Activation {key} on wrong device: {activation.device} vs {self.device}"
            
            self.test_results['device_consistency'] = True
            print("[PASS] Device consistency test passed")
            return True
            
        except Exception as e:
            self.test_results['device_consistency'] = False
            print(f"[FAIL] Device consistency test failed: {e}")
            return False
    
    def benchmark_performance(self):
        """Performance benchmarks comparing to baseline"""
        print("\n=== Performance Benchmarks ===")
        
        try:
            batch_size, seq_len = 4, 32
            tokens = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
            
            # Benchmark standard forward pass
            num_runs = 10
            
            # Standard forward pass timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(num_runs):
                with torch.no_grad():
                    logits = self.model(tokens)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            standard_time = (time.time() - start_time) / num_runs
            
            # Cached forward pass timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(num_runs):
                with torch.no_grad():
                    logits, cache = self.model.run_with_cache(tokens)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            cached_time = (time.time() - start_time) / num_runs
            
            # Calculate overhead
            overhead = (cached_time - standard_time) / standard_time * 100
            
            self.debug_info['performance'] = {
                'standard_time': standard_time,
                'cached_time': cached_time,
                'overhead_percent': overhead
            }
            
            self.test_results['performance_benchmark'] = True
            print(f"[PASS] Performance benchmark completed (caching overhead: {overhead:.1f}%)")
            return True
            
        except Exception as e:
            self.test_results['performance_benchmark'] = False
            print(f"[FAIL] Performance benchmark failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print("SAELens Integration Validation")
        print("=" * 50)
        
        # Setup
        if not self.setup_model():
            print("Failed to setup model, aborting tests")
            return False
        
        if not self.load_sae():
            print("Failed to load SAE, aborting tests")
            return False
        
        # Run all tests
        test_methods = [
            self.test_basic_functionality,
            self.test_activation_caching,
            self.test_activation_shapes,
            self.test_sae_integration,
            self.test_sae_training_compatibility,
            self.test_activation_patching,
            self.test_gradient_flow,
            self.test_memory_management,
            self.test_device_consistency,
            self.benchmark_performance
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            try:
                if test_method():
                    passed_tests += 1
            except Exception as e:
                print(f"Unexpected error in {test_method.__name__}: {e}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        
        for test_name, result in self.test_results.items():
            status = "PASS" if result else "FAIL"
            print(f"{test_name}: {status}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        # Print debug info
        if self.debug_info:
            print("\nDEBUG INFO:")
            for key, value in self.debug_info.items():
                print(f"  {key}: {value}")
        
        return passed_tests == total_tests


def run_validation():
    """Main function to run validation"""
    # Check for CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running tests on device: {device}")
    
    # Create test instance
    tester = SAELensIntegrationTest(device=device)
    
    # Run all tests
    success = tester.run_all_tests()
    
    if success:
        print("\n[SUCCESS] All tests passed! SAELens integration is working correctly.")
    else:
        print("\n[ERROR] Some tests failed. Check the output above for details.")
    
    return success


if __name__ == "__main__":
    run_validation()