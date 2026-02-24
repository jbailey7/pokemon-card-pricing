"""
Shared constants used across all model modules.
"""

# ImageNet normalisation — standard for pretrained torchvision models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Model input size (EfficientNet-B0 default)
INPUT_SIZE = 224

# Embedding dimensionality output by the projection head
EMBEDDING_DIM = 512
