<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Customer Review Sentiment Analysis - Copilot Instructions

This is a customer review sentiment analysis project built with PyTorch and modern NLP techniques. The project follows the methodology from bentrevett/pytorch-sentiment-analysis but is optimized for customer review data.

## Project Context

- **Primary Goal**: Analyze sentiment in customer reviews using multiple deep learning approaches
- **Target Domain**: Customer reviews (products, services, restaurants, etc.)
- **Framework**: PyTorch with torchtext, transformers, and datasets libraries
- **Models**: Neural Bag of Words, LSTM, CNN, and Transformer-based architectures

## Code Style Guidelines

### Model Implementation
- Follow PyTorch best practices for model definition
- Use `nn.Module` for all models with proper `__init__` and `forward` methods
- Include detailed docstrings explaining model architecture and parameters
- Use type hints for function parameters and return values

### Data Processing
- Use datasets library for data loading and preprocessing
- Implement proper tokenization with spaCy or transformers tokenizers
- Handle variable-length sequences with padding and attention masks
- Include data validation and error handling

### Training Pipeline
- Implement separate training and evaluation functions
- Use proper device handling (CPU/GPU)
- Include progress bars with tqdm
- Save model checkpoints and training metrics
- Implement early stopping and learning rate scheduling

### Code Organization
- Keep models in separate files under `models/` directory
- Use utility functions for common operations
- Implement proper logging and error handling
- Follow consistent naming conventions

## Key Libraries and Usage

### Core Dependencies
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from transformers import AutoTokenizer, AutoModel
import datasets
import numpy as np
import matplotlib.pyplot as plt
```

### Model Patterns
- Use bidirectional LSTMs for better context understanding
- Apply dropout for regularization
- Use pre-trained embeddings (GloVe, Word2Vec) for better initialization
- For transformers, use `AutoTokenizer` and `AutoModel` from transformers library

### Customer Review Specific Considerations
- Handle informal language and typos in reviews
- Consider review length variations (short vs. long reviews)
- Account for domain-specific vocabulary
- Implement proper text cleaning while preserving sentiment indicators

## Performance Optimization
- Use batch processing for inference
- Implement proper memory management for large datasets
- Use DataLoader with appropriate batch sizes
- Consider mixed precision training for faster training

## Testing and Validation
- Implement comprehensive evaluation metrics (accuracy, precision, recall, F1)
- Use cross-validation for robust performance estimates
- Include visualization of training curves and confusion matrices
- Test on diverse review types and lengths

## Business Application Focus
- Provide interpretable results with confidence scores
- Include batch processing capabilities for business use
- Generate actionable insights from sentiment analysis
- Consider real-time inference requirements
