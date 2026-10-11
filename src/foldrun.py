import os, time, random, torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
import dataset, loss, model as M, evalscale

BASE = '/content/drive/MyDrive/depth_project'
ALL = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']

DEFAULT = dict(variant='full', use_flow=True, separable=True, ch=48,
               lam_dyn=2.0, lam_tex=1.0, lam_t=1.0, lam_si=0.85,
               epochs=10, batch=4, lr=1e-3, crop=(256, 512), seed=0,
               max_steps=None, eval_stride=1, monitor_every=5)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def to_cuda(b):
    return {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}

def save_atomic(obj, path):
    torch.save(obj, path + '.tmp')
    os.replace(path + '.tmp', path)

def predict_fp32(net):
    def f(b):
        return net(b['dav2'], b['flow']).exp(), net(b['dav2_next'], b['flow_next']).exp()
    return f

def run_fold(test_scene, **cfg):
    c = {**DEFAULT, **cfg}
    train_scenes = [s for s in ALL if s != test_scene]
    name = f"{c['variant']}_test{test_scene[-2:]}_s{c['seed']}"
    ck_dir = f'{BASE}/checkpoints/folds/{name}'
    res_dir = f'{BASE}/results/folds'
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    final_path = f'{res_dir}/{name}.csv'
    if os.path.exists(final_path):
        print(f'[{name}] already finished, skipping')
        return pd.read_csv(final_path)

    dataset.SPLITS['fold_test'] = [test_scene]
    set_seed(c['seed'])
    net = M.FlowConditionedStudent(ch=c['ch'], separable=c['separable'], use_flow=c['use_flow']).cuda()
    opt = torch.optim.Adam(net.parameters(), c['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=c['epochs'], eta_min=c['lr'] * 0.05)
    scaler = torch.amp.GradScaler('cuda')
    start, hist = 0, []

    last = f'{ck_dir}/last.pt'
    if os.path.exists(last):
        st = torch.load(last, map_location='cuda', weights_only=False)
        net.load_state_dict(st['net'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        scaler.load_state_dict(st['scaler'])
        start, hist = st['epoch'], st['hist']
        print(f'[{name}] resuming after epoch {start}')

    print(f'[{name}] parameters {net.count_params()} | train on {train_scenes} | test on {test_scene}')
    train_ds = dataset.DepthPairs(scenes=train_scenes, crop=c['crop'])

    for ep in range(start, c['epochs']):
        set_seed(c['seed'] * 1000 + ep)
        dl = DataLoader(train_ds, batch_size=c['batch'], shuffle=True, num_workers=2, drop_last=True)
        net.train()
        t0 = time.time()
        acc, n = np.zeros(3), 0
        for i, b in enumerate(dl):
            if c['max_steps'] and i >= c['max_steps']:
                break
            b = to_cuda(b)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                o0 = net(b['dav2'], b['flow'])
                o1 = net(b['dav2_next'], b['flow_next']) if c['lam_t'] > 0 else None
            o1f = o1.float() if o1 is not None else None
            tot, si, te = loss.total_loss(o0.float(), o1f, b, c['lam_dyn'], c['lam_tex'], c['lam_t'], c['lam_si'])
            scaler.scale(tot).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            acc += [tot.item(), si.item(), te.item()]
            n += 1
        sched.step()
        row = dict(epoch=ep + 1, train_loss=acc[0] / n, train_silog=acc[1] / n,
                   train_temporal=acc[2] / n, minutes=(time.time() - t0) / 60)
        hist.append(row)
        save_atomic({'net': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                     'scaler': scaler.state_dict(), 'epoch': ep + 1, 'hist': hist, 'cfg': str(c)}, last)
        msg = f"[{name}] epoch {ep+1:2d}/{c['epochs']} | train loss {row['train_loss']:.4f} | {row['minutes']:.1f} min"
        if (ep + 1) % c['monitor_every'] == 0 and ep + 1 < c['epochs']:
            net.eval()
            m = evalscale.score_scaled(predict_fp32(net), 'fold_test', 8)
            net.train()
            msg += f" | check on {test_scene}: scaled {m['AbsRel_scaled']:.4f} dyn {m['dyn_scaled']:.4f} flicker {m['flickerLog']:.4f}"
        print(msg)

    net.eval()
    m = evalscale.score_scaled(predict_fp32(net), 'fold_test', c['eval_stride'])
    out = pd.DataFrame([{**dict(variant=c['variant'], test_scene=test_scene, seed=c['seed'],
                                epochs=c['epochs'], params=net.count_params()), **m}])
    out.to_csv(final_path, index=False)
    print(f'[{name}] FINAL on {test_scene}: scaled {m["AbsRel_scaled"]:.4f} | dyn {m["dyn_scaled"]:.4f} '
          f'| tex {m["tex_scaled"]:.4f} | flicker {m["flickerLog"]:.4f} (dyn {m["flickerLogDyn"]:.4f})')
    return out
