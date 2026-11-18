import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils
from alignment.osbp import ImageClassifier as Classifier
from models.MAOSDAN_shift import MAOSDAN_Shift


class OG_MoE(nn.Module):
    def __init__(self, arch, num_classes, bottleneck_dim=256, shared_backbone=True,
                 pi_u=0.5, shift_kwargs=None):
        super().__init__()
        self.arch = arch
        self.num_classes = int(num_classes)
        self.num_known = self.num_classes - 1
        self.shared = bool(shared_backbone)
        self.pi_u = float(pi_u)
        shift_kwargs = dict(shift_kwargs or {})

        if self.shared:
            backbone = utils.get_model(arch)
            self.backbone = backbone
            self.expertA = Classifier(backbone, num_classes, bottleneck_dim=bottleneck_dim)
            self.expertB = MAOSDAN_Shift(backbone, num_classes,
                                         bottleneck_dim=bottleneck_dim, **shift_kwargs)
        else:
            self.backbone = None
            self.expertA = Classifier(utils.get_model(arch), num_classes,
                                      bottleneck_dim=bottleneck_dim)
            self.expertB = MAOSDAN_Shift(utils.get_model(arch), num_classes,
                                         bottleneck_dim=bottleneck_dim, **shift_kwargs)

        gate_in = 2 * self.num_classes + 5
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward_all(self, x):
        if self.shared:
            f0 = self.backbone(x)
            fA = self.expertA.bottleneck(f0)
            fB = self.expertB.bottleneck(f0)
        else:
            fA = self.expertA.bottleneck(self.expertA.backbone(x))
            fB = self.expertB.bottleneck(self.expertB.backbone(x))
        return {
            'fA': fA,
            'fB': fB,
            'logitsA': self.expertA.head(fA),
            'logitsB': self.expertB.head(fB),
        }

    def forward_a(self, x):
        if self.shared:
            fA = self.expertA.bottleneck(self.backbone(x))
        else:
            fA = self.expertA.bottleneck(self.expertA.backbone(x))
        return fA, self.expertA.head(fA)

    def fuse(self, logitsA, logitsB, pi_u=None):
        pA = F.softmax(logitsA.float(), dim=1)
        pB = F.softmax(logitsB.float(), dim=1)
        entA = -(pA * torch.log(pA.clamp_min(1e-12))).sum(1, keepdim=True)
        entB = -(pB * torch.log(pB.clamp_min(1e-12))).sum(1, keepdim=True)
        maxA = pA.max(1, keepdim=True).values
        maxB = pB.max(1, keepdim=True).values
        pi = float(self.pi_u if pi_u is None else pi_u)
        pi_col = torch.full_like(entA, pi)
        inp = torch.cat([pA.detach(), pB.detach(), entA.detach(), entB.detach(),
                         maxA.detach(), maxB.detach(), pi_col], dim=1)
        g = self.gate(inp)
        fused = g * pA + (1.0 - g) * pB
        return fused, g, pA, pB

    def forward(self, x):
        out = self.forward_all(x)
        fused, _g, _pA, _pB = self.fuse(out['logitsA'], out['logitsB'])
        return fused

    def forward_eval(self, x):
        out = self.forward_all(x)
        fused, _g, _pA, _pB = self.fuse(out['logitsA'], out['logitsB'])
        return out['fB'], torch.log(fused.clamp_min(1e-12))

    def get_parameters(self):
        groups, seen = [], set()

        def add(params, lr):
            ps = [p for p in params if id(p) not in seen]
            for p in ps:
                seen.add(id(p))
            if ps:
                groups.append({'params': ps, 'lr': float(lr)})

        if self.shared:
            add(self.backbone.parameters(), 0.1)
        else:
            add(self.expertA.backbone.parameters(), 0.1)
            add(self.expertB.backbone.parameters(), 0.1)
        add(self.expertA.bottleneck.parameters(), 1.0)
        add(self.expertA.head.parameters(), 1.0)
        add(self.expertB.bottleneck.parameters(), 1.0)
        add(self.expertB.head.parameters(), 1.0)
        add(self.gate.parameters(), 1.0)
        return groups

    def set_pi_u(self, pi_u):
        self.pi_u = float(pi_u)

    def set_shift_epoch(self, epoch, pi_u):
        self.set_pi_u(pi_u)
        if hasattr(self.expertB, 'set_shift_epoch'):
            self.expertB.set_shift_epoch(epoch, pi_u)

    def observe_target(self, features, logits):
        if hasattr(self.expertB, 'observe_target'):
            self.expertB.observe_target(features, logits)

    def shift_loss(self, features, logits):
        if hasattr(self.expertB, 'shift_loss'):
            return self.expertB.shift_loss(features, logits)
        return None

    def on_shift_epoch_end(self):
        if hasattr(self.expertB, 'on_shift_epoch_end'):
            return self.expertB.on_shift_epoch_end()
        return ''

    def shift_describe(self):
        if hasattr(self.expertB, 'shift_describe'):
            return self.expertB.shift_describe()
        return ''
