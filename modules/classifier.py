from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch

__all__ = ['Classifier']


class Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True, pool_layer=None):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)

        f = self.bottleneck(f)
        predictions = self.head(f)
        if self.training:
            return predictions, f
        else:
            return predictions

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params


class ImageClassifier(Classifier):
    pass
