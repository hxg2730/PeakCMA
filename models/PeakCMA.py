"""PeakCMA equations (7)-(8); labels never enter forward()."""
import torch
from torch import nn
from models.backbone import CrossModalBackbone


class PeakCMA(nn.Module):
    def __init__(self, peak_channels, **kwargs):
        super().__init__()
        self.backbone = CrossModalBackbone(**kwargs)
        width, horizon = self.backbone.channel, self.backbone.pred_len
        mask = torch.zeros(self.backbone.num_nodes)
        mask[list(peak_channels)] = 1
        self.register_buffer('peak_mask', mask)
        self.horizon_embedding = nn.Embedding(horizon, width)
        self.features = nn.Sequential(nn.Linear(3 * width, 64), nn.GELU(), nn.Dropout(.1))
        self.localisation = nn.Linear(64, 1)
        self.correction = nn.Sequential(nn.Linear(65, 64), nn.GELU(), nn.Dropout(.1), nn.Linear(64, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, x, mark, embeddings, return_aux=False):
        base, z = self.backbone(x, mark, embeddings, return_features=True)
        b, h, n = base.shape
        z = z[:, None].expand(-1, h, -1, -1)
        e = self.horizon_embedding.weight[None, :, None].expand(b, -1, n, -1)
        f = self.features(torch.cat((z, e, z * e), dim=-1))
        logits = self.localisation(f).squeeze(-1)
        score = logits.sigmoid()
        residual = self.correction(torch.cat((f, base[..., None]), dim=-1)).squeeze(-1)
        delta = self.peak_mask * score * residual
        prediction = base + delta
        if return_aux:
            return prediction, {'base': base, 'score': score, 'logits': logits, 'residual': residual, 'delta': delta}
        return prediction

    def freeze_backbone(self, freeze=True):
        self.backbone.requires_grad_(not freeze)
        if freeze:
            self.backbone.eval()

    def load_baseline(self, state):
        self.backbone.load_state_dict(state, strict=True)

    def param_num(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
