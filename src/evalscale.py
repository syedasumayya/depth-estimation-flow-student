import torch
from torch.utils.data import DataLoader, Subset
import dataset, loss
from evalbase import cuda

@torch.no_grad()
def score_scaled(depth_fn, split='val', stride=1):
    ds = dataset.DepthPairs(split)
    sub = Subset(ds, list(range(0, len(ds), stride)))
    dl = DataLoader(sub, batch_size=1, shuffle=False, num_workers=2)
    S = {k: [0.0, 0] for k in ['raw', 'ms', 'ms_dyn', 'ms_tex', 'bias', 'tl', 'tl_dyn']}

    def add(k, x, m):
        S[k][0] += x[m].sum().item()
        S[k][1] += int(m.sum().item())

    for b in dl:
        b = cuda(b)
        d0, d1 = depth_fn(b)
        d0 = d0.clamp(min=0.1, max=80.0)
        d1 = d1.clamp(min=0.1, max=80.0)
        v = b['valid'] > 0
        lr = torch.log(d0 / b['gt'])
        add('raw', (d0 - b['gt']).abs() / b['gt'], v)
        add('bias', lr, v)
        d0s = d0 * torch.exp(-lr[v].median())            # one scale per frame, from the median
        rel = (d0s - b['gt']).abs() / b['gt']
        add('ms', rel, v)
        add('ms_dyn', rel, v & (b['dyn'] > 0))
        add('ms_tex', rel, v & (b['tex'] > 0))

        l1w, inside = loss.warp(torch.log(d1), b['flow'])
        g1w, _ = loss.warp(b['gt_next'], b['flow'], 'nearest')
        v1w, _ = loss.warp(b['valid_next'], b['flow'], 'nearest')
        vt = (b['valid'] * v1w * b['flow_ok'] * inside) > 0
        change_pred = torch.log(d0) - l1w
        change_true = torch.log(b['gt'].clamp(min=1e-3)) - torch.log(g1w.clamp(min=1e-3))
        e = (change_pred - change_true).abs()
        add('tl', e, vt)
        add('tl_dyn', e, vt & (b['dyn'] > 0))

    r = lambda k: S[k][0] / max(S[k][1], 1)
    return {'AbsRel_raw': r('raw'), 'AbsRel_scaled': r('ms'), 'dyn_scaled': r('ms_dyn'),
            'tex_scaled': r('ms_tex'), 'logbias': r('bias'),
            'flickerLog': r('tl'), 'flickerLogDyn': r('tl_dyn')}
