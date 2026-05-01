# Atria Registry

<div align="center">

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

*A powerful and flexible registry system for managing and organizing machine learning components in the Atria ecosystem.*

</div>

## 🚀 Overview

Atria Registry is a centralized component management system designed to streamline the organization, registration, and retrieval of machine learning components including datasets, models, transformations, pipelines, and more. Built with Hydra integration, it provides a robust configuration management framework for complex ML workflows.

### Key Features

- **🔧 Centralized Management**: Single point of access for all ML components
- **📦 Modular Design**: Organized registry groups for different component types
- **⚙️ Hydra Integration**: Seamless configuration management with Hydra framework
- **🔍 Type Safety**: Full type annotations and validation support
- **🎯 Easy Registration**: Simple decorators and methods for component registration
- **🧩 Extensible**: Support for custom registry groups and components
- **📚 Rich Documentation**: Comprehensive docstrings and examples

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Registry Groups](#registry-groups)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 📦 Installation

### Using pip

```bash
pip install atria-registry
```

### Using uv (recommended)

```bash
uv add atria-registry
```

### Development Installation

```bash
git clone https://github.com/saifullah3396/atria_registry.git
uv sync --dev
```

## 🚀 Quick Start

### Basic Usage

```python
from atria_registry import MODEL, DATA_TRANSFORM, DATASET

# Register a new model
@MODEL.register("my_custom_model")
class MyCustomModel:
    def __init__(self, hidden_dim: int = 256):
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        # Model implementation
        return x

# Register a data transformation
@DATA_TRANSFORM.register("my_transform")
class MyTransform:
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
    
    def __call__(self, data):
        # Transform implementation
        return data

# Load registered components
model_config = MODEL.load_config("my_custom_model")
model = MODEL.load_from_registry("my_custom_model", hidden_dim=512)
```

### Using with Hydra Configurations

```python
from hydra import compose, initialize
from atria_registry import MODEL

# Initialize Hydra
with initialize(config_path="configs"):
    cfg = compose(config_name="model_config")
    
# Load model from configuration
model = MODEL.load_from_registry(
    cfg.model.name,
    overrides=["model.hidden_dim=1024"]
)
```

## 📊 Registry Groups

Atria Registry organizes components into specialized registry groups:

| Registry Group | Purpose | Examples |
|----------------|---------|----------|
| **DATASET** | Dataset components | Custom datasets, data loaders |
| **DATA_PIPELINE** | Data processing pipelines | ETL workflows, preprocessing |
| **DATA_TRANSFORM** | Data transformations | Normalization, augmentation |
| **BATCH_SAMPLER** | Batch sampling strategies | Custom samplers, balancing |
| **MODEL** | Machine learning models | Neural networks, classifiers |
| **MODEL_PIPELINE** | Complete model pipelines | End-to-end inference |
| **TASK_PIPELINE** | Task-specific workflows | Training, evaluation |
| **METRIC** | Evaluation metrics | Accuracy, F1-score, custom metrics |
| **LR_SCHEDULER** | Learning rate schedulers | Step, cosine, exponential |
| **OPTIMIZER** | Optimization algorithms | Adam, SGD, custom optimizers |
| **ENGINE** | Training/inference engines | Trainers, evaluators |

## 💡 Usage Examples

### Registering Components

#### Simple Registration

```python
from atria_registry import DATA_TRANSFORM

@DATA_TRANSFORM.register("standardize")
class StandardizeTransform:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
```

#### Batch Registration

```python
from atria_registry import MODEL

# Register multiple models at once
MODEL.register_modules(
    module_paths=[
        "my_package.models.ResNet",
        "my_package.models.VGG",
        "my_package.models.DenseNet"
    ],
    module_names=["resnet", "vgg", "densenet"]
)
```

#### Custom Registry Groups

```python
from atria_registry.registry_group import RegistryGroup

# Create custom registry group
CUSTOM_PROCESSORS = RegistryGroup(name="custom_processors")

@CUSTOM_PROCESSORS.register("my_processor")
class MyProcessor:
    def process(self, data):
        return data
```

### Loading and Configuration

#### Basic Loading

```python
from atria_registry import MODEL

# Load with default parameters
model = MODEL.load_from_registry("resnet")

# Load with custom parameters
model = MODEL.load_from_registry(
    "resnet", 
    num_classes=10,
    pretrained=True
)
```

#### Advanced Configuration

```python
from atria_registry import MODEL

# Load configuration only
config = MODEL.load_config("resnet")

# Load with overrides
model = MODEL.load_from_registry(
    "resnet",
    overrides=[
        "num_classes=100",
        "dropout=0.5",
        "optimizer.lr=0.001"
    ]
)

# Get configuration with overrides
config = MODEL.load_config(
    "resnet",
    overrides={"num_classes": 100, "dropout": 0.5}
)
```

### Torchvision Integration

```python
from atria_registry import DATA_TRANSFORM

# Register torchvision transforms
DATA_TRANSFORM.register_torchvision_transform(
    "RandomHorizontalFlip",
    p=0.5
)

# Use the registered transform
transform = DATA_TRANSFORM.load_from_registry("RandomHorizontalFlip")
```

## ⚙️ Configuration

### Registry Configuration

```python
from atria_registry.module_registry import ModuleRegistry

# Access the central registry
registry = ModuleRegistry()

# Register all lazy modules
registry.register_all_modules()

# Get specific registry group
model_group = registry.get_registry_group("MODEL")
```

### Hydra Integration

```yaml
# config.yaml
defaults:
  - model: resnet
  - optimizer: adam
  - _self_

model:
  num_classes: 10
  dropout: 0.1

optimizer:
  lr: 0.001
  weight_decay: 1e-4
```

### Writing Configurations to YAML

```python
from atria_registry.utilities import write_registry_to_yaml

# Export all configurations to YAML files
write_registry_to_yaml("./configs")
```

## 📖 API Reference

### Core Classes

#### `ModuleRegistry`
Central singleton registry managing all component groups.

```python
registry = ModuleRegistry()
registry.register_all_modules()
group = registry.get_registry_group("MODEL")
```

#### `RegistryGroup`
Base class for managing component registrations.

```python
group = RegistryGroup(name="custom", is_factory=True)
group.register_module("my_module", lazy_build=True)
component = group.load_from_registry("my_module")
```

#### `RegistryEntry`
Represents individual registered components.

```python
entry = RegistryEntry(
    group="model",
    module_name="resnet",
    module_path="torchvision.models.resnet18"
)
entry.register()
```

### Specialized Groups

- **`DatasetRegistryGroup`**: Enhanced dataset management
- **`DataTransformRegistryGroup`**: Transform-specific features
- **`ModelRegistryGroup`**: Model pattern matching

## 🛠️ Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/saifullah3396/atria_registry.git

# Install with development dependencies
uv sync --dev

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
./scripts/test.sh

# Run with coverage
uv run pytest --cov=atria_registry

# Run specific test file
uv run pytest tests/test_module_registry.py
```

### Code Quality

```bash
# Format code
./scripts/format.sh

# Lint code
./scripts/lint.sh

# Type checking
uv run mypy src/atria_registry
```

### Project Structure

```
atria_registry/
├── src/atria_registry/
│   ├── __init__.py              # Main exports and registry groups
│   ├── constants.py             # Registry constants
│   ├── module_registry.py       # Central registry manager
│   ├── registry_config_mixin.py # Configuration mixin
│   ├── registry_entry.py        # Individual entry management
│   ├── registry_group.py        # Registry group implementations
│   └── utilities.py             # Helper utilities
├── tests/                       # Test suite
├── scripts/                     # Development scripts
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Contribution Guide

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow [PEP 8](https://pep8.org/) style guidelines
- Add tests for new features
- Update documentation for API changes
- Use type hints throughout the codebase
- Write descriptive commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 🙏 Acknowledgments

- Built with [Hydra](https://hydra.cc/) for configuration management
- Powered by [hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/) for configuration building
- Developed by the Atria Development Team

## 📞 Support

- **Documentation**: [Full API Documentation](https://atria-docs.example.com)
- **Issues**: [GitHub Issues](https://github.com/saifullah3396/atria_registry/issues)
- **Discussions**: [GitHub Discussions](https://github.com/saifullah3396/atria_registry/discussions)

---

<div align="center">
Made with ❤️ by the Atria Development Team
</div>