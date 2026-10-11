import torch
from torch.utils.data import DataLoader, Subset
import dataset, loss

def cuda(b):
    return {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}

def fit_log_mapping(split, stride=10):
    """Least squares: log(true depth) = c0 + c1 * normalized teacher output, over a split."""
    ds = dataset.DepthPairs(split)
    n = sx = sy = sxx = sxy = 0.0
    for i in range(0, len(ds), stride):
        s = ds[i]
        v = s['valid'][0] > 0
        x = s['dav2'][0][v].double()
        y = torch.log(s['gt'][0][v]).double()
        n += x.numel()
        sx += x.sum().item(); sy += y.sum().item()
        sxx += (x * x).sum().item(); sxy += (x * y).sum().item()
    c1 = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    c0 = (sy - c1 * sx) / n
    return c0, c1

def fixed_fn(c0, c1):
    def f(b):
        return torch.exp(c0 + c1 * b['dav2']), torch.exp(c0 + c1 * b['dav2_next'])
    return f

def _fit_inv(p, g, v):
    """Per-frame fit in inverse-depth space (uses the answer key)."""
    m = v[0] > 0
    x = p[0][m].double()
    y = (1.0 / g[0][m]).double()
    n = x.numel()
    sx, sy, sxx, sxy = x.sum(), y.sum(), (x * x).sum(), (x * y).sum()
    a = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    b = (sy - a * sx) / n
    return 1.0 / (a * p + b).clamp(min=1.0 / 80.0)

def perframe_fn():
    def f(b):
        return (_fit_inv(b['dav2'], b['gt'], b['valid']),
                _fit_inv(b['dav2_next'], b['gt_next'], b['valid_next']))
    return f

@torch.no_grad()
def score(depth_fn, split='val', stride=3):
    ds = dataset.DepthPairs(split)
    sub = Subset(ds, list(range(0, len(ds), stride)))
    dl = DataLoader(sub, batch_size=1, shuffle=False, num_workers=2)
    S = {k: [0.0, 0] for k in ['all', 'dyn', 'tex', 't', 'td', 'bias', 'so']}

    def add(k, x, m):
        S[k][0] += x[m].sum().item()
        S[k][1] += int(m.sum().item())

    for b in dl:
        b = cuda(b)
        d0, d1 = depth_fn(b)
        d0 = d0.clamp(max=80.0)
        d1 = d1.clamp(max=80.0)
        v = b['valid'] > 0
        rel = (d0 - b['gt']).abs() / b['gt']
        add('all', rel, v)
        add('dyn', rel, v & (b['dyn'] > 0))
        add('tex', rel, v & (b['tex'] > 0))
        lr = torch.log(d0 / b['gt'])
        add('bias', lr, v)
        d0s = d0 * torch.exp(-lr[v].median())
        add('so', (d0s - b['gt']).abs() / b['gt'], v)
        d1w, inside = loss.warp(d1, b['flow'])
        g1w, _ = loss.warp(b['gt_next'], b['flow'], 'nearest')
        v1w, _ = loss.warp(b['valid_next'], b['flow'], 'nearest')
        vt = (b['valid'] * v1w * b['flow_ok'] * inside) > 0
        e = ((d0 - d1w) - (b['gt'] - g1w)).abs() / b['gt'].clamp(min=1e-3)
        add('t', e, vt)
        add('td', e, vt & (b['dyn'] > 0))

    r = lambda k: S[k][0] / max(S[k][1], 1)
    return {'AbsRel_all': r('all'), 'AbsRel_dyn': r('dyn'), 'AbsRel_tex': r('tex'),
            'tempErr': r('t'), 'tempErrDyn': r('td'),
            'logbias': r('bias'), 'AbsRel_ifScaleFixed': r('so')}
