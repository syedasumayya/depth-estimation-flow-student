import random, numpy as np, torch
from torch.utils.data import Dataset

CACHE = '/content/cache'
FAST = '/content/cache/fast'

SPLITS = {
    'train': ['Scene01', 'Scene18', 'Scene20'],
    'val':   ['Scene02'],
    'test':  ['Scene06'],
}
N_FRAMES = {'Scene01': 447, 'Scene02': 233, 'Scene06': 270, 'Scene18': 339, 'Scene20': 837}

def normalize(p):
    """Bring the teacher's output to a common range WITHOUT using ground truth."""
    med = np.median(p)
    spread = np.mean(np.abs(p - med))
    return (p - med) / (spread + 1e-6)

def unpack(m):
    return (m & 1) > 0, (m & 2) > 0, (m & 4) > 0

def to_t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).float()

class DepthPairs(Dataset):
    """One example = frame t and frame t+1 (with the flow of BOTH frames).
    Pass crop=(h, w) for random crops in training."""
    def __init__(self, split='train', scenes=None, crop=None):
        self.scenes = scenes if scenes is not None else SPLITS[split]
        self.crop = crop
        self.items = [(sc, fr) for sc in self.scenes for fr in range(N_FRAMES[sc] - 2)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sc, fr = self.items[i]
        p0 = normalize(np.load(f'{CACHE}/{sc}/{fr:05d}.npy').astype(np.float32))
        p1 = normalize(np.load(f'{CACHE}/{sc}/{fr+1:05d}.npy').astype(np.float32))
        a = np.load(f'{FAST}/{sc}/{fr:05d}.npz')
        b = np.load(f'{FAST}/{sc}/{fr+1:05d}.npz')
        g0 = a['gt'].astype(np.float32) / 100.0
        g1 = b['gt'].astype(np.float32) / 100.0
        fl = a['flow'].astype(np.float32)
        fl1 = b['flow'].astype(np.float32)
        fok = a['fok'] > 0
        v0, d0, t0 = unpack(np.load(f'{CACHE}/masks/{sc}/{fr:05d}.npy'))
        v1, _, _ = unpack(np.load(f'{CACHE}/masks/{sc}/{fr+1:05d}.npy'))
        g0 = np.where(v0, g0, 1.0)
        g1 = np.where(v1, g1, 1.0)

        if self.crop is not None:
            H, W = p0.shape
            ch, cw = self.crop
            top = random.randint(0, H - ch)
            left = random.randint(0, W - cw)
            crop_slice = (slice(top, top + ch), slice(left, left + cw))
            arrays = [p0, p1, g0, g1, fl, fl1, fok, v0, v1, d0, t0]
            p0, p1, g0, g1, fl, fl1, fok, v0, v1, d0, t0 = [x[crop_slice] for x in arrays]

        return {
            'dav2': to_t(p0)[None], 'dav2_next': to_t(p1)[None],
            'flow': to_t(fl).permute(2, 0, 1), 'flow_next': to_t(fl1).permute(2, 0, 1),
            'gt': to_t(g0)[None], 'gt_next': to_t(g1)[None],
            'valid': to_t(v0)[None], 'valid_next': to_t(v1)[None],
            'dyn': to_t(d0)[None], 'tex': to_t(t0)[None],
            'flow_ok': to_t(fok)[None],
            'scene': sc, 'frame': fr,
        }
