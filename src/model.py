import torch
import torch.nn as nn

class Block(nn.Module):
    """One residual block: output = input + small correction."""
    def __init__(self, ch, dilation, separable=True):
        super().__init__()
        if separable:
            self.body = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, groups=ch),  # look around, per channel
                nn.SiLU(),
                nn.Conv2d(ch, ch, 1),                                                   # mix the channels
                nn.SiLU())
        else:
            self.body = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation),             # regular filter
                nn.SiLU())

    def forward(self, x):
        return x + self.body(x)

class FlowConditionedStudent(nn.Module):
    """Input: normalized DAv2 depth (B,1,H,W) and flow (B,2,H,W). Output: log-depth (B,1,H,W)."""
    def __init__(self, ch=48, dilations=(1, 2, 4, 8, 16, 1, 2, 4, 8, 16),
                 separable=True, use_flow=True, flow_scale=30.0):
        super().__init__()
        self.use_flow, self.flow_scale = use_flow, flow_scale
        self.stem = nn.Sequential(nn.Conv2d(3 if use_flow else 1, ch, 3, padding=1), nn.SiLU())
        self.blocks = nn.Sequential(*[Block(ch, d, separable) for d in dilations])
        self.head = nn.Conv2d(ch, 1, 1)
        nn.init.normal_(self.head.weight, std=1e-3)      # correction starts almost at zero
        nn.init.zeros_(self.head.bias)
        self.c0 = nn.Parameter(torch.tensor(2.8))        # starting guess: log(16 m)
        self.c1 = nn.Parameter(torch.tensor(-0.5))       # closer (bigger teacher value) -> smaller depth

    def forward(self, dav2, flow):
        x = torch.cat([dav2, flow / self.flow_scale], 1) if self.use_flow else dav2
        base = self.c0 + self.c1 * dav2
        return base + self.head(self.blocks(self.stem(x)))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
