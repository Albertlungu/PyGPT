# Configuration Manager - Build Guide

## Overview
The Configuration Manager centralizes all hyperparameters, paths, and settings for your PyGPT project. This makes it easy to experiment with different configurations, share settings between components, and maintain consistency across your codebase.

**One place to control everything!**

---

## Why Use a Config System?

### Problems Without Config:

❌ Hyperparameters scattered across multiple files
❌ Hard to track which settings produced which results
❌ Difficult to reproduce experiments
❌ Error-prone when changing settings

### Benefits With Config:

✅ All settings in one place
✅ Easy to experiment with different configurations
✅ Reproducible experiments
✅ Can save/load configurations
✅ Type checking and validation

---

## What to Store in Config

### Model Architecture Settings:

- [ ] **vocab_size**: Size of vocabulary from tokenizer
- [ ] **embedding_dim**: Dimension of token embeddings (e.g., 256, 512)
- [ ] **max_seq_length**: Maximum sequence length (e.g., 128, 512)
- [ ] **num_layers**: Number of transformer blocks (e.g., 4, 6, 12)
- [ ] **num_heads**: Number of attention heads (e.g., 8, 12)
- [ ] **ff_dim**: Feed-forward hidden dimension (e.g., 1024, 2048)
- [ ] **dropout_rate**: Dropout probability (optional, e.g., 0.1)

### Training Hyperparameters:

- [ ] **learning_rate**: Optimizer learning rate (e.g., 3e-4)
- [ ] **batch_size**: Sequences per batch (e.g., 16, 32)
- [ ] **num_epochs**: Training epochs (e.g., 5, 10)
- [ ] **gradient_clip_value**: Max gradient norm (e.g., 1.0)
- [ ] **weight_decay**: L2 regularization (optional, e.g., 0.01)

### Data Settings:

- [ ] **train_data_path**: Path to training text file
- [ ] **val_data_path**: Path to validation text file
- [ ] **tokenizer_path**: Path to saved tokenizer (e.g., "artifacts/tokenizer.pkl")
- [ ] **train_split**: Fraction for training (e.g., 0.9)
- [ ] **val_split**: Fraction for validation (e.g., 0.1)

### Paths and Directories:

- [ ] **checkpoint_dir**: Where to save model checkpoints
- [ ] **log_dir**: Where to save training logs
- [ ] **output_dir**: Where to save generated text
- [ ] **artifacts_dir**: Where to save embeddings, tokenizer, etc.

### Training Control:

- [ ] **eval_interval**: Iterations between validation (e.g., 500)
- [ ] **save_interval**: Iterations between checkpoints (e.g., 1000)
- [ ] **log_interval**: Iterations between logging (e.g., 100)
- [ ] **max_iterations**: Total training iterations (alternative to epochs)

### Generation Settings:

- [ ] **temperature**: Sampling temperature (e.g., 0.8, 1.0)
- [ ] **top_k**: Top-k sampling parameter (e.g., 40)
- [ ] **top_p**: Nucleus sampling parameter (e.g., 0.9)
- [ ] **max_gen_length**: Max tokens to generate (e.g., 100)

### Miscellaneous:

- [ ] **seed**: Random seed for reproducibility (e.g., 42)
- [ ] **device**: "cpu" or "cuda" (for future GPU support)
- [ ] **num_workers**: Data loading workers (for future parallelization)

---

## Implementation Approach 1: Simple Dictionary

### Basic Config Class:

```python
class Config:
    """Simple configuration using class attributes."""

    # Model Architecture
    vocab_size = 10000
    embedding_dim = 256
    max_seq_length = 128
    num_layers = 4
    num_heads = 8
    ff_dim = 1024

    # Training
    learning_rate = 3e-4
    batch_size = 16
    num_epochs = 5
    gradient_clip_value = 1.0

    # Data
    train_data_path = "data/train.txt"
    tokenizer_path = "artifacts/tokenizer.pkl"
    train_split = 0.9

    # Paths
    checkpoint_dir = "checkpoints/"
    log_dir = "logs/"

    # Training Control
    eval_interval = 500
    save_interval = 1000
    log_interval = 100

    # Generation
    temperature = 0.8
    max_gen_length = 100

    # Misc
    seed = 42
```

### Usage:

```python
from src.utils.config import Config

# Access settings
print(f"Learning rate: {Config.learning_rate}")

# Use in code
embedding_layer = EmbeddingLayer(
    vocab_size=Config.vocab_size,
    embedding_dim=Config.embedding_dim,
    max_seq_length=Config.max_seq_length
)
```

---

## Implementation Approach 2: Dictionary-Based

### Config as Dictionary:

```python
def get_default_config():
    """Returns default configuration as dictionary."""
    return {
        # Model Architecture
        'vocab_size': 10000,
        'embedding_dim': 256,
        'max_seq_length': 128,
        'num_layers': 4,
        'num_heads': 8,
        'ff_dim': 1024,

        # Training
        'learning_rate': 3e-4,
        'batch_size': 16,
        'num_epochs': 5,
        'gradient_clip_value': 1.0,

        # Data
        'train_data_path': 'data/train.txt',
        'tokenizer_path': 'artifacts/tokenizer.pkl',
        'train_split': 0.9,

        # Paths
        'checkpoint_dir': 'checkpoints/',
        'log_dir': 'logs/',

        # Training Control
        'eval_interval': 500,
        'save_interval': 1000,
        'log_interval': 100,

        # Generation
        'temperature': 0.8,
        'max_gen_length': 100,

        # Misc
        'seed': 42
    }
```

### Usage:

```python
config = get_default_config()

# Access settings
print(f"Learning rate: {config['learning_rate']}")

# Modify for experiments
config['learning_rate'] = 1e-4
config['batch_size'] = 32
```

---

## Implementation Approach 3: Advanced (Recommended)

### Config Class with Validation:

```python
import json
import os
from typing import Optional

class ModelConfig:
    """Configuration class with validation and save/load."""

    def __init__(self, **kwargs):
        # Model Architecture
        self.vocab_size = kwargs.get('vocab_size', 10000)
        self.embedding_dim = kwargs.get('embedding_dim', 256)
        self.max_seq_length = kwargs.get('max_seq_length', 128)
        self.num_layers = kwargs.get('num_layers', 4)
        self.num_heads = kwargs.get('num_heads', 8)
        self.ff_dim = kwargs.get('ff_dim', 1024)

        # Training
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.batch_size = kwargs.get('batch_size', 16)
        self.num_epochs = kwargs.get('num_epochs', 5)
        self.gradient_clip_value = kwargs.get('gradient_clip_value', 1.0)

        # Data
        self.train_data_path = kwargs.get('train_data_path', 'data/train.txt')
        self.tokenizer_path = kwargs.get('tokenizer_path', 'artifacts/tokenizer.pkl')
        self.train_split = kwargs.get('train_split', 0.9)

        # Paths
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'checkpoints/')
        self.log_dir = kwargs.get('log_dir', 'logs/')

        # Training Control
        self.eval_interval = kwargs.get('eval_interval', 500)
        self.save_interval = kwargs.get('save_interval', 1000)
        self.log_interval = kwargs.get('log_interval', 100)

        # Generation
        self.temperature = kwargs.get('temperature', 0.8)
        self.max_gen_length = kwargs.get('max_gen_length', 100)

        # Misc
        self.seed = kwargs.get('seed', 42)

        # Validate after initialization
        self.validate()

    def validate(self):
        """Validate configuration values."""
        assert self.embedding_dim > 0, "embedding_dim must be positive"
        assert self.embedding_dim % self.num_heads == 0, \
            "embedding_dim must be divisible by num_heads"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.batch_size > 0, "batch_size must be positive"
        assert 0 < self.train_split < 1, "train_split must be between 0 and 1"
        assert self.temperature > 0, "temperature must be positive"

    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def save(self, filepath):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def __repr__(self):
        """String representation of config."""
        lines = ["ModelConfig:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
```

---

## Usage Examples

### Create Default Config:

```python
config = ModelConfig()
print(config)
```

### Create Custom Config:

```python
config = ModelConfig(
    embedding_dim=512,
    num_layers=6,
    learning_rate=1e-4,
    batch_size=32
)
```

### Save Config:

```python
config.save('configs/experiment_1.json')
```

### Load Config:

```python
config = ModelConfig.load('configs/experiment_1.json')
```

### Use in Training:

```python
from src.utils.config import ModelConfig

# Load config
config = ModelConfig.load('configs/my_experiment.json')

# Initialize model components
embedding_layer = EmbeddingLayer(
    vocab_size=config.vocab_size,
    embedding_dim=config.embedding_dim,
    max_seq_length=config.max_seq_length
)

transformer_blocks = [
    TransformerBlock(
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim
    )
    for _ in range(config.num_layers)
]

# Train with config
train(
    model_components=model_components,
    learning_rate=config.learning_rate,
    batch_size=config.batch_size,
    num_epochs=config.num_epochs
)
```

---

## Predefined Configs for Different Scales

### Tiny Model (for testing):

```python
def get_tiny_config():
    return ModelConfig(
        vocab_size=5000,
        embedding_dim=128,
        max_seq_length=64,
        num_layers=2,
        num_heads=4,
        ff_dim=512,
        batch_size=8
    )
```

### Small Model:

```python
def get_small_config():
    return ModelConfig(
        vocab_size=10000,
        embedding_dim=256,
        max_seq_length=128,
        num_layers=4,
        num_heads=8,
        ff_dim=1024,
        batch_size=16
    )
```

### Medium Model:

```python
def get_medium_config():
    return ModelConfig(
        vocab_size=30000,
        embedding_dim=512,
        max_seq_length=256,
        num_layers=6,
        num_heads=8,
        ff_dim=2048,
        batch_size=16
    )
```

### Large Model (requires significant compute):

```python
def get_large_config():
    return ModelConfig(
        vocab_size=50000,
        embedding_dim=768,
        max_seq_length=512,
        num_layers=12,
        num_heads=12,
        ff_dim=3072,
        batch_size=8
    )
```

---

## Experiment Tracking

### Track Multiple Experiments:

```python
# Experiment 1: Low learning rate
config1 = ModelConfig(learning_rate=1e-4)
config1.save('configs/exp1_lr1e-4.json')

# Experiment 2: High learning rate
config2 = ModelConfig(learning_rate=1e-3)
config2.save('configs/exp2_lr1e-3.json')

# Experiment 3: Larger model
config3 = ModelConfig(embedding_dim=512, num_layers=6)
config3.save('configs/exp3_large.json')
```

### Load Best Config After Experiments:

```python
# Load the config that performed best
best_config = ModelConfig.load('configs/exp2_lr1e-3.json')
```

---

## Environment-Specific Configs

### Development Config (fast iteration):

```python
def get_dev_config():
    return ModelConfig(
        embedding_dim=128,
        num_layers=2,
        batch_size=8,
        max_seq_length=64,
        eval_interval=100,
        save_interval=500
    )
```

### Production Config (quality):

```python
def get_prod_config():
    return ModelConfig(
        embedding_dim=512,
        num_layers=6,
        batch_size=32,
        max_seq_length=256,
        num_epochs=10
    )
```

---

## Command-Line Integration

### Parse Args and Override Config:

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    # Load config or create default
    if args.config:
        config = ModelConfig.load(args.config)
    else:
        config = ModelConfig()

    # Override with command-line args
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_epochs:
        config.num_epochs = args.num_epochs

    print(config)
    # Continue with training...
```

### Usage:

```bash
# Use default config
python train.py

# Use saved config
python train.py --config configs/experiment_1.json

# Override specific values
python train.py --config configs/experiment_1.json --learning_rate 1e-4
```

---

## Testing Checklist

### Validation Tests:

- [ ] Test invalid embedding_dim (not divisible by num_heads)
- [ ] Test negative learning_rate
- [ ] Test train_split > 1.0
- [ ] Verify validation catches all errors

### Save/Load Tests:

- [ ] Save config to file
- [ ] Load config from file
- [ ] Verify loaded config matches saved config
- [ ] Test with different file paths

### Usage Tests:

- [ ] Initialize model components with config
- [ ] Modify config and verify changes
- [ ] Use in training loop

---

## Best Practices

### DO:

✅ Use descriptive config names (e.g., `exp1_high_lr.json`)
✅ Save config with every experiment
✅ Version control your configs
✅ Document why you chose specific values
✅ Validate all config values

### DON'T:

❌ Hard-code hyperparameters in code
❌ Reuse config files across incompatible experiments
❌ Forget to save config before long training runs
❌ Mix configs from different model versions

---

## Config File Organization

```
configs/
├── default.json           # Default config
├── experiments/
│   ├── exp1_baseline.json
│   ├── exp2_large_lr.json
│   ├── exp3_big_model.json
│   └── exp4_long_seq.json
├── models/
│   ├── tiny.json
│   ├── small.json
│   ├── medium.json
│   └── large.json
└── production/
    └── best_model.json
```

---

## Success Criteria

You've successfully built the Config system when:

- [ ] All hyperparameters in one place
- [ ] Can save and load configs easily
- [ ] Validation prevents invalid configurations
- [ ] Easy to create new experiment configs
- [ ] Can track which config produced which results
- [ ] Config integrates smoothly with training code
- [ ] Can override config values from command line

---

## Next Steps

After implementing Config:
1. Create default config file
2. Update training code to use config
3. Create configs for different experiments
4. Test save/load functionality
5. Document your config choices
6. Use config versioning for reproducibility
