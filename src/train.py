import os, time, random, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader, Subset
import dataset, loss, model as M, evalscale

BASE = '/content/drive/MyDrive/depth_project'

DEFAULT = dict(name='full', use_flow=True, separable=True, ch=48,
               lam_dyn=2.0, lam_tex=1.0, lam_t=1.0, lam_si=0.85,
               epochs=20, batch=4, lr=1e-3, crop=(256, 512), seed=0,
               max_steps=None, val_stride=3)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def cuda(b):
    return {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}

@torch.no_grad()
def evaluate(net, split='val', stride=3):
    """Score the student on a split (full frames). Same metrics as the baseline tables."""
    was_training = net.training
    net.eval()
    ds = dataset.DepthPairs(split)
    sub = Subset(ds, list(range(0, len(ds), stride)))
    dl = DataLoader(sub, batch_size=1, shuffle=False, num_workers=2)
    S = {k: [0.0, 0] for k in ['all', 'dyn', 'tex', 't', 'td']}

    def add(k, x, m):
        S[k][0] += x[m].sum().item()
        S[k][1] += int(m.sum().item())

    for b in dl:
        b = cuda(b)
        d0 = net(b['dav2'], b['flow']).exp().clamp(max=80.0)
        d1 = net(b['dav2_next'], b['flow_next']).exp().clamp(max=80.0)
        v = b['valid'] > 0
        rel = (d0 - b['gt']).abs() / b['gt']
        add('all', rel, v)
        add('dyn', rel, v & (b['dyn'] > 0))
        add('tex', rel, v & (b['tex'] > 0))
        d1w, inside = loss.warp(d1, b['flow'])
        g1w, _ = loss.warp(b['gt_next'], b['flow'], 'nearest')
        v1w, _ = loss.warp(b['valid_next'], b['flow'], 'nearest')
        vt = (b['valid'] * v1w * b['flow_ok'] * inside) > 0
        e = ((d0 - d1w) - (b['gt'] - g1w)).abs() / b['gt'].clamp(min=1e-3)
        add('t', e, vt)
        add('td', e, vt & (b['dyn'] > 0))

    if was_training:
        net.train()
    r = lambda k: S[k][0] / max(S[k][1], 1)
    return {'AbsRel_all': r('all'), 'AbsRel_dyn': r('dyn'), 'AbsRel_tex': r('tex'),
            'tempErr': r('t'), 'tempErrDyn': r('td')}

def save_atomic(obj, path):
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)

def run(cfg):
    c = {**DEFAULT, **cfg}
    name = f"{c['name']}_s{c['seed']}"
    ck_dir = f'{BASE}/checkpoints/{name}'
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(f'{BASE}/results/runs', exist_ok=True)
    log_path = f'{BASE}/results/runs/{name}_log.csv'

    set_seed(c['seed'])
    net = M.FlowConditionedStudent(ch=c['ch'], separable=c['separable'], use_flow=c['use_flow']).cuda()
    opt = torch.optim.Adam(net.parameters(), c['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c['epochs'], eta_min=c['lr'] * 0.05)
    start, best, hist = 0, float('inf'), []

    last = f'{ck_dir}/last.pt'
    if os.path.exists(last):
        st = torch.load(last, map_location='cuda', weights_only=False)
        net.load_state_dict(st['net']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
        start, best, hist = st['epoch'], st['best'], st['hist']
        print(f'[{name}] resuming after epoch {start}')
    if start >= c['epochs']:
        print(f'[{name}] already finished, skipping'); return hist

    print(f'[{name}] parameters: {net.count_params()} | epochs {start + 1}..{c["epochs"]}')
    train_ds = dataset.DepthPairs('train', crop=c['crop'])

    for ep in range(start, c['epochs']):
        set_seed(c['seed'] * 1000 + ep)
        dl = DataLoader(train_ds, batch_size=c['batch'], shuffle=True, num_workers=2, drop_last=True)
        net.train(); t0 = time.time(); acc = np.zeros(3); n = 0
        for i, b in enumerate(dl):
            if c['max_steps'] and i >= c['max_steps']:
                break
            b = cuda(b)
            opt.zero_grad()
            o0 = net(b['dav2'], b['flow'])
            o1 = net(b['dav2_next'], b['flow_next']) if c['lam_t'] > 0 else None
            tot, si, te = loss.total_loss(o0, o1, b, c['lam_dyn'], c['lam_tex'], c['lam_t'], c['lam_si'])
            tot.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            acc += [tot.item(), si.item(), te.item()]; n += 1
        lr_now = opt.param_groups[0]['lr']
        sched.step()

        net.eval()
        m = evalscale.score_scaled(lambda b: (net(b['dav2'], b['flow']).exp(), net(b['dav2_next'], b['flow_next']).exp()), 'val', c['val_stride'])
        net.train()
        row = dict(epoch=ep + 1, train_loss=acc[0] / n, train_silog=acc[1] / n, train_temporal=acc[2] / n,
                   lr=lr_now, minutes=(time.time() - t0) / 60, **m)
        hist.append(row)
        is_best = m['AbsRel_scaled'] < best
        if is_best:
            best = m['AbsRel_scaled']
            save_atomic({'net': net.state_dict(), 'epoch': ep + 1, 'cfg': str(c)}, f'{ck_dir}/best.pt')
        save_atomic({'net': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                     'epoch': ep + 1, 'best': best, 'hist': hist}, last)
        pd.DataFrame(hist).to_csv(log_path, index=False)
        print(f"[{name}] epoch {ep+1:2d}/{c['epochs']} | train {row['train_loss']:.4f} | "
              f"val scaled all {m['AbsRel_scaled']:.4f} dyn {m['dyn_scaled']:.4f} tex {m['tex_scaled']:.4f} (raw {m['AbsRel_raw']:.3f}) | "
              f"flicker {m['flickerLog']:.4f} (dyn {m['flickerLogDyn']:.4f}) | {row['minutes']:.1f} min{'  <- best' if is_best else ''}")
    return hist

def load_best(name, cfg=None):
    c = {**DEFAULT, **(cfg or {})}
    net = M.FlowConditionedStudent(ch=c['ch'], separable=c['separable'], use_flow=c['use_flow']).cuda()
    st = torch.load(f'{BASE}/checkpoints/{name}/best.pt', map_location='cuda', weights_only=False)
    net.load_state_dict(st['net'])
    return net.eval()
