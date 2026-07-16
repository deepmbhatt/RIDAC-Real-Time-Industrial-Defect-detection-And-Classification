"""Model definitions and preprocessing used by the trained RIDAC checkpoint."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    efficientnet_b0,
    mobilenet_v3_large,
)


SUPPORTED_BACKBONES = ("mobilenet_v3_large", "efficientnet_b0")


def build_feature_extractor(backbone: str, pretrained: bool):
    if backbone == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        network = mobilenet_v3_large(weights=weights)
        return network.features, 960
    if backbone == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        network = efficientnet_b0(weights=weights)
        return network.features, 1280
    raise ValueError(
        f"Unsupported backbone {backbone!r}; choose from {SUPPORTED_BACKBONES}"
    )


class ClassConditionedClassifier(nn.Module):
    """Backbone with class-guided attention and FiLM conditioning."""

    def __init__(
        self,
        num_classes: int,
        backbone: str,
        pretrained: bool = True,
        embedding_dim: int = 128,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.features, feature_dim = build_feature_extractor(backbone, pretrained)
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        self.attention_query = nn.Linear(embedding_dim, feature_dim)
        self.film = nn.Linear(embedding_dim, feature_dim * 2)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, images: torch.Tensor, class_ids: torch.Tensor):
        feature_map = self.features(images)
        embedding = self.class_embedding(class_ids)
        keys = F.normalize(feature_map, dim=1)
        query = F.normalize(self.attention_query(embedding), dim=1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        attention_logits = torch.einsum("bchw,bc->bhw", keys, query) * scale
        attention = attention_logits.flatten(1).softmax(dim=1).view_as(
            attention_logits
        )
        attended = torch.einsum("bchw,bhw->bc", feature_map, attention)
        pooled = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        gamma, beta = self.film(embedding).chunk(2, dim=1)
        conditioned = pooled * (1.0 + torch.tanh(gamma)) + beta
        logits = self.classifier(
            torch.cat((attended, conditioned), dim=1)
        ).squeeze(1)
        return logits, attention


def build_model(
    backbone: str,
    num_classes: int,
    device: torch.device,
    pretrained: bool = True,
    embedding_dim: int = 128,
    dropout: float = 0.30,
) -> ClassConditionedClassifier:
    model = ClassConditionedClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )
    return model.to(device, memory_format=torch.channels_last)


class SquarePad:
    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        return ImageOps.expand(
            image,
            border=(left, top, side - width - left, side - height - top),
            fill=0,
        )


def build_transforms(image_size: int):
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8, fill=0),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform
