from .alignment_dataset import (
    AlignmentDatasetGenerator,
    AlignmentExample,
    BehaviorType,
    PromptType,
    create_alignment_dataset
)

from .sae_trainer import (
    AlignmentSAETrainer,
    AlignmentSAEConfig,
    AlignmentActivationCollector,
    create_alignment_sae_config
)

__version__ = "0.1.0"

__all__ = [
    # Dataset components
    "AlignmentDatasetGenerator",
    "AlignmentExample", 
    "BehaviorType",
    "PromptType",
    "create_alignment_dataset",
    
    # Training components
    "AlignmentSAETrainer",
    "AlignmentSAEConfig",
    "AlignmentActivationCollector",
    "create_alignment_sae_config",
]