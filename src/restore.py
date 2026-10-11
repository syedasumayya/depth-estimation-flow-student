import os, sys, glob, time, shutil, subprocess
import numpy as np, cv2
from tqdm.auto import tqdm

BASE = '/content/drive/MyDrive/depth_project'
DATA = '/content/data'
CACHE = '/content/cache'
TARS = f'{BASE}/tars'
SCENES = ['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20']
sys.path.insert(0, f'{BASE}/src')

def n_frames(sc):
    return len(glob.glob(f'{DATA}/{sc}/clone/frames/rgb/Camera_0/*'))

def restore_data():
    os.makedirs(DATA, exist_ok=True)
    marker = f'{DATA}/.done_textgt'
    if not os.path.exists(marker):
        fname = 'vkitti_2.0.3_textgt.tar.gz'
        shutil.copy(f'{TARS}/{fname}', f'/content/{fname}')
        subprocess.run(['tar', '-xf', f'/content/{fname}', '-C', DATA], check=True)
        os.remove(f'/content/{fname}')
        open(marker, 'w').close()
    for name in ['rgb', 'depth', 'forwardFlow', 'classSegmentation', 'instanceSegmentation']:
        marker = f'{DATA}/.done_{name}_clone'
        if os.path.exists(marker):
            continue
        tar = f'{TARS}/{name}_clone.tar'
        if not os.path.exists(tar):
            raise FileNotFoundError(f'Drive backup missing: {tar}')
        subprocess.run(['tar', '-xf', tar, '-C', DATA], check=True)
        open(marker, 'w').close()

def restore_teacher():
    os.makedirs(CACHE, exist_ok=True)
    need = []
    for sc in SCENES:
        out = f'{CACHE}/{sc}'
        if len(glob.glob(f'{out}/*.npy')) == n_frames(sc):
            continue
        tar = f'{TARS}/dav2cache_{sc}.tar'
        if os.path.exists(tar):
            subprocess.run(['tar', '-xf', tar, '-C', CACHE], check=True)
        else:
            need.append(sc)
    if not need:
        return
    import torch
    try:
        import transformers
    except ImportError:
        subprocess.run(['pip', 'install', '-q', 'transformers'])
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    mid = 'depth-anything/Depth-Anything-V2-Small-hf'
    processor = AutoImageProcessor.from_pretrained(mid)
    net = AutoModelForDepthEstimation.from_pretrained(mid).to('cuda').eval()
    for sc in need:
        out = f'{CACHE}/{sc}'
        os.makedirs(out, exist_ok=True)
        r = f'{DATA}/{sc}/clone/frames'
        for fr in tqdm(range(n_frames(sc)), desc=f'teacher {sc}'):
            if os.path.exists(f'{out}/{fr:05d}.npy'):
                continue
            rgb = cv2.cvtColor(cv2.imread(f'{r}/rgb/Camera_0/rgb_{fr:05d}.jpg'), cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                inp = processor(images=rgb, return_tensors='pt').to('cuda')
                pred = net(**inp).predicted_depth
                pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=rgb.shape[:2],
                                                       mode='bicubic', align_corners=False)[0, 0]
            np.save(f'{out}/{fr:05d}.npy', pred.cpu().numpy().astype(np.float16))

def restore_masks():
    import masks
    for sc in SCENES:
        out = f'{CACHE}/masks/{sc}'
        os.makedirs(out, exist_ok=True)
        n = n_frames(sc)
        if len(glob.glob(f'{out}/*.npy')) == n:
            continue
        r = f'{DATA}/{sc}/clone/frames'
        for fr in tqdm(range(n), desc=f'masks {sc}'):
            rgb = cv2.cvtColor(cv2.imread(f'{r}/rgb/Camera_0/rgb_{fr:05d}.jpg'), cv2.COLOR_BGR2RGB)
            gt = cv2.imread(f'{r}/depth/Camera_0/depth_{fr:05d}.png',
                            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32) / 100
            valid = masks.valid_depth_mask(gt) & (gt <= 80)
            dyn = masks.dynamic_mask(sc, fr) & valid
            tex = masks.textureless_mask(rgb) & valid
            packed = valid.astype(np.uint8) | (dyn.astype(np.uint8) << 1) | (tex.astype(np.uint8) << 2)
            np.save(f'{out}/{fr:05d}.npy', packed)

def restore_fast():
    for sc in SCENES:
        out = f'{CACHE}/fast/{sc}'
        os.makedirs(out, exist_ok=True)
        n = n_frames(sc)
        if len(glob.glob(f'{out}/*.npz')) == n:
            continue
        r = f'{DATA}/{sc}/clone/frames'
        flows = sorted(glob.glob(f'{r}/forwardFlow/Camera_0/*'))
        for fr in tqdm(range(n), desc=f'fast {sc}'):
            gt = cv2.imread(f'{r}/depth/Camera_0/depth_{fr:05d}.png',
                            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if fr < len(flows):
                bgr = cv2.imread(flows[fr], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                h, w, _ = bgr.shape
                bad = bgr[..., 0] == 0
                fl = 2.0 / (2**16 - 1.0) * bgr[..., 2:0:-1].astype(np.float32) - 1
                fl[..., 0] *= (w - 1)
                fl[..., 1] *= (h - 1)
                fl[bad] = 0
                fok = (~bad).astype(np.uint8)
            else:
                fl = np.zeros(gt.shape + (2,), np.float32)
                fok = np.zeros(gt.shape, np.uint8)
            np.savez(f'{out}/{fr:05d}.npz', gt=gt, flow=fl.astype(np.float16), fok=fok)

def restore_all():
    for name, fn in [('data', restore_data), ('teacher outputs', restore_teacher),
                     ('masks', restore_masks), ('fast cache', restore_fast)]:
        t0 = time.time()
        fn()
        print(f'[{name}] ready ({(time.time() - t0) / 60:.1f} min)')
    for sc in SCENES:
        n = n_frames(sc)
        c = [len(glob.glob(f'{CACHE}/{p}/{sc}/*')) for p in ('', 'masks', 'fast')]
        c[0] = len(glob.glob(f'{CACHE}/{sc}/*.npy'))
        print(f'{sc}: frames {n} | teacher {c[0]} | masks {c[1]} | fast {c[2]}')
