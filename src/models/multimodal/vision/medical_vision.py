"""
Medical Vision Model for Multimodal AI

Specialized vision encoder for medical images including X-rays, CT scans,
MRIs, and other medical imaging modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0, vit_b_16
import timm

from ....utils.logging.logger import get_logger

logger = get_logger(__name__)


class MedicalImageProcessor:
    """
    Medical image preprocessing and augmentation.
    
    Handles various medical imaging modalities with appropriate
    preprocessing pipelines for each type.
    """
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        
        # Define transforms for different medical imaging modalities
        self.transforms = {
            "xray": self._get_xray_transforms(),
            "ct": self._get_ct_transforms(), 
            "mri": self._get_mri_transforms(),
            "ultrasound": self._get_ultrasound_transforms(),
            "pathology": self._get_pathology_transforms(),
            "dermatology": self._get_dermatology_transforms(),
            "generic": self._get_generic_transforms()
        }
    
    def _get_xray_transforms(self) -> transforms.Compose:
        """Transforms for chest X-rays."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Medical-specific augmentations
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])
    
    def _get_ct_transforms(self) -> transforms.Compose:
        """Transforms for CT scans."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            # CT-specific normalization (Hounsfield units)
            transforms.Lambda(lambda x: torch.clamp(x, -1000, 1000) / 1000.0),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def _get_mri_transforms(self) -> transforms.Compose:
        """Transforms for MRI scans."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def _get_ultrasound_transforms(self) -> transforms.Compose:
        """Transforms for ultrasound images."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def _get_pathology_transforms(self) -> transforms.Compose:
        """Transforms for pathology images."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Pathology-specific augmentations
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
    
    def _get_dermatology_transforms(self) -> transforms.Compose:
        """Transforms for dermatology images."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
    
    def _get_generic_transforms(self) -> transforms.Compose:
        """Generic transforms for unknown medical image types."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def process_image(self, image: torch.Tensor, modality: str = "generic") -> torch.Tensor:
        """Process medical image based on modality."""
        transform = self.transforms.get(modality, self.transforms["generic"])
        return transform(image)


class MedicalVisionEncoder(nn.Module):
    """
    Medical vision encoder with multiple backbone options.
    
    Supports various architectures optimized for medical imaging:
    - ResNet variants
    - EfficientNet
    - Vision Transformer
    - Medical-specific architectures
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        hidden_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.backbone_name = backbone
        self.hidden_dim = hidden_dim
        
        # Initialize backbone
        self.backbone = self._create_backbone(backbone, pretrained)
        
        # Get backbone output dimension
        backbone_dim = self._get_backbone_dim()
        
        # Projection layer to hidden dimension
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # Optional classification head
        if num_classes:
            self.classifier = nn.Linear(hidden_dim, num_classes)
        else:
            self.classifier = None
        
        logger.info(f"Medical vision encoder initialized with {backbone} backbone")
    
    def _create_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """Create vision backbone."""
        if backbone == "resnet50":
            model = resnet50(pretrained=pretrained)
            # Remove final classification layer
            model = nn.Sequential(*list(model.children())[:-1])
            
        elif backbone == "efficientnet_b0":
            model = efficientnet_b0(pretrained=pretrained)
            model.classifier = nn.Identity()
            
        elif backbone == "vit_b_16":
            model = vit_b_16(pretrained=pretrained)
            model.heads = nn.Identity()
            
        elif backbone == "medical_resnet":
            # Custom medical ResNet
            model = self._create_medical_resnet()
            
        else:
            # Use timm for additional models
            try:
                model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
            except:
                logger.warning(f"Unknown backbone {backbone}, using ResNet50")
                model = resnet50(pretrained=pretrained)
                model = nn.Sequential(*list(model.children())[:-1])
        
        return model
    
    def _create_medical_resnet(self) -> nn.Module:
        """Create medical-specific ResNet architecture."""
        # Custom ResNet with medical imaging optimizations
        model = resnet50(pretrained=True)
        
        # Modify first layer for medical images
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Add attention mechanisms
        model.layer4 = nn.Sequential(
            model.layer4,
            SpatialAttention(),
            ChannelAttention(2048)
        )
        
        # Remove final layers
        model = nn.Sequential(*list(model.children())[:-2])
        
        return model
    
    def _get_backbone_dim(self) -> int:
        """Get output dimension of backbone."""
        if self.backbone_name == "resnet50":
            return 2048
        elif self.backbone_name == "efficientnet_b0":
            return 1280
        elif self.backbone_name == "vit_b_16":
            return 768
        elif self.backbone_name == "medical_resnet":
            return 2048
        else:
            # Try to infer from model
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = self.backbone(dummy_input)
                return output.shape[-1]
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through vision encoder."""
        # Extract features
        features = self.backbone(x)
        
        # Global average pooling if needed
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
        
        # Project to hidden dimension
        projected = self.projection(features)
        
        outputs = {
            "features": features,
            "projected_features": projected
        }
        
        # Classification if available
        if self.classifier:
            logits = self.classifier(projected)
            outputs["logits"] = logits
        
        return outputs


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for medical images."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention mechanism for medical images."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = avg_out + max_out
        attention = self.sigmoid(attention)
        return x * attention


class MedicalImageClassifier(nn.Module):
    """
    Medical image classifier for specific tasks.
    
    Can be used for:
    - Disease classification
    - Anatomical structure detection
    - Abnormality detection
    - Multi-label classification
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        multilabel: bool = False,
        class_names: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        
        # Vision encoder
        self.encoder = MedicalVisionEncoder(
            backbone=backbone,
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        # Loss function
        if multilabel:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        outputs = self.encoder(x)
        
        if self.multilabel:
            # Apply sigmoid for multilabel
            probs = torch.sigmoid(outputs["logits"])
        else:
            # Apply softmax for single-label
            probs = F.softmax(outputs["logits"], dim=-1)
        
        outputs["probabilities"] = probs
        return outputs
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
        """Make predictions with interpretable output."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            probs = outputs["probabilities"]
            
            if self.multilabel:
                predictions = (probs > threshold).float()
                predicted_classes = []
                for i in range(probs.shape[0]):
                    sample_classes = [
                        self.class_names[j] for j in range(self.num_classes)
                        if predictions[i, j] == 1
                    ]
                    predicted_classes.append(sample_classes)
            else:
                predictions = torch.argmax(probs, dim=-1)
                predicted_classes = [
                    self.class_names[pred.item()] for pred in predictions
                ]
            
            return {
                "predictions": predictions,
                "probabilities": probs,
                "predicted_classes": predicted_classes,
                "confidence": torch.max(probs, dim=-1)[0] if not self.multilabel else torch.mean(probs, dim=-1)
            }


class MedicalMultiModalFusion(nn.Module):
    """
    Fusion module for combining medical images with text.
    
    Combines visual features from medical images with textual
    information (symptoms, history, etc.) for comprehensive analysis.
    """
    
    def __init__(
        self,
        vision_dim: int = 768,
        text_dim: int = 768,
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        
        # Project to common dimension
        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        
        # Cross-attention for fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(fusion_dim, fusion_dim)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Fuse vision and text features."""
        batch_size = vision_features.shape[0]
        
        # Project to common dimension
        vision_proj = self.vision_proj(vision_features)
        text_proj = self.text_proj(text_features)
        
        # Add sequence dimension if needed
        if len(vision_proj.shape) == 2:
            vision_proj = vision_proj.unsqueeze(1)
        if len(text_proj.shape) == 2:
            text_proj = text_proj.unsqueeze(1)
        
        # Cross-attention: vision attends to text
        vision_attended, _ = self.cross_attention(
            query=vision_proj,
            key=text_proj,
            value=text_proj,
            key_padding_mask=text_mask
        )
        
        # Cross-attention: text attends to vision
        text_attended, _ = self.cross_attention(
            query=text_proj,
            key=vision_proj,
            value=vision_proj,
            key_padding_mask=vision_mask
        )
        
        # Concatenate attended features
        fused_features = torch.cat([
            vision_attended.squeeze(1),
            text_attended.squeeze(1)
        ], dim=-1)
        
        # Apply fusion layers
        fused_output = self.fusion_layers(fused_features)
        final_output = self.output_proj(fused_output)
        
        return {
            "fused_features": final_output,
            "vision_attended": vision_attended.squeeze(1),
            "text_attended": text_attended.squeeze(1)
        }


def create_medical_vision_model(
    task: str,
    num_classes: int,
    backbone: str = "resnet50",
    **kwargs
) -> nn.Module:
    """
    Factory function to create medical vision models.
    
    Args:
        task: Type of task ("classification", "detection", "segmentation")
        num_classes: Number of output classes
        backbone: Vision backbone architecture
        **kwargs: Additional model parameters
        
    Returns:
        Configured medical vision model
    """
    if task == "classification":
        return MedicalImageClassifier(
            num_classes=num_classes,
            backbone=backbone,
            **kwargs
        )
    elif task == "multimodal":
        return MedicalMultiModalFusion(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")


# Example usage and testing
if __name__ == "__main__":
    # Test medical vision encoder
    encoder = MedicalVisionEncoder(backbone="resnet50")
    
    # Test input
    x = torch.randn(2, 3, 224, 224)
    outputs = encoder(x)
    
    print(f"Features shape: {outputs['features'].shape}")
    print(f"Projected features shape: {outputs['projected_features'].shape}")
    
    # Test medical image classifier
    classifier = MedicalImageClassifier(
        num_classes=5,
        class_names=["Normal", "Pneumonia", "COVID-19", "Tuberculosis", "Other"]
    )
    
    predictions = classifier.predict(x)
    print(f"Predictions: {predictions['predicted_classes']}")
    print(f"Confidence: {predictions['confidence']}")
