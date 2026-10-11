import torch
import torch.nn.functional as F

def region_weights(dyn, tex, lam_dyn=2.0, lam_tex=1.0):
    """Normal pixel = 1, moving object += lam_dyn, flat area += lam_tex."""
    return 1.0 + lam_dyn * dyn + lam_tex * tex

def silog(pred_log, gt, valid, w, lam=0.85):
    """Weighted SILog, computed per image and averaged over the batch."""
    d = pred_log - torch.log(gt.clamp(min=1e-3))
    m = valid * w
    n = m.sum(dim=(1, 2, 3)).clamp(min=1.0)
    mean_d = (m * d).sum(dim=(1, 2, 3)) / n
    mean_d2 = (m * d * d).sum(dim=(1, 2, 3)) / n
    per_image = torch.sqrt((mean_d2 - lam * mean_d ** 2).clamp(min=1e-6))
    return per_image.mean()

def warp(x, flow, mode='bilinear'):
    """Read x at (pixel + flow). Also returns 1 where that location is inside the image."""
    B, _, H, W = x.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=x.device),
                            torch.arange(W, device=x.device), indexing='ij')
    px = xs[None].float() + flow[:, 0]
    py = ys[None].float() + flow[:, 1]
    inside = ((px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)).float()[:, None]
    grid = torch.stack([2 * px / (W - 1) - 1, 2 * py / (H - 1) - 1], dim=-1)
    out = F.grid_sample(x, grid, mode=mode, padding_mode='border', align_corners=True)
    return out, inside

def temporal(pred0, pred1, batch, w):
    """Does the predicted change between frames match the true change?"""
    flow = batch['flow']
    p1w, inside = warp(pred1, flow)
    g1w, _ = warp(batch['gt_next'], flow, 'nearest')
    v1w, _ = warp(batch['valid_next'], flow, 'nearest')
    v = batch['valid'] * v1w * batch['flow_ok'] * inside
    change_pred = pred0 - p1w
    change_true = torch.log(batch['gt'].clamp(min=1e-3)) - torch.log(g1w.clamp(min=1e-3))
    e = (change_pred - change_true).abs()
    return (v * w * e).sum() / (v * w).sum().clamp(min=1.0)

def total_loss(out0, out1, batch, lam_dyn=2.0, lam_tex=1.0, lam_t=1.0, lam_si=0.85):
    """out0/out1 = student log-depth for frame t / t+1 (out1 may be None if lam_t == 0)."""
    w = region_weights(batch['dyn'], batch['tex'], lam_dyn, lam_tex)
    l_si = silog(out0, batch['gt'], batch['valid'], w, lam_si)
    l_t = temporal(out0, out1, batch, w) if lam_t > 0 else out0.new_zeros(())
    return l_si + lam_t * l_t, l_si.detach(), l_t.detach()
