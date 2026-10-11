import glob, cv2, numpy as np, pandas as pd

DATA = '/content/data'
_cache = {}

def load_inst_code(scene, frame, data=DATA):
    f = f'{data}/{scene}/clone/frames/instanceSegmentation/Camera_0/instancegt_{frame:05d}.png'
    im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype(np.uint32)
    return (im[..., 0] << 16) | (im[..., 1] << 8) | im[..., 2]

def load_bbox(scene, data=DATA):
    key = ('bbox', scene)
    if key not in _cache:
        b = pd.read_csv(f'{data}/{scene}/clone/bbox.txt', sep=' ')
        _cache[key] = b[b.cameraID == 0].reset_index(drop=True)
    return _cache[key]

def track_colors(scene, data=DATA):
    """trackID -> color code, learned once per scene by pixel-count matching."""
    key = ('colors', scene)
    if key in _cache:
        return _cache[key]
    b = load_bbox(scene, data)
    colors = {}
    for fr in sorted(b.frame.unique()):
        rows = b[b.frame == fr]
        todo = [r for _, r in rows.iterrows() if int(r.trackID) not in colors]
        if not todo:
            continue
        code = load_inst_code(scene, int(fr), data)
        cols, cnts = np.unique(code, return_counts=True)
        for r in todo:
            cand = cols[cnts == r.number_pixels]
            if len(cand) == 1:
                colors[int(r.trackID)] = int(cand[0])
    _cache[key] = colors
    return colors

def dynamic_mask(scene, frame, data=DATA):
    """True where a pixel belongs to an object flagged isMoving in this frame."""
    b = load_bbox(scene, data)
    mv = b[(b.frame == frame) & b.isMoving]
    cols = track_colors(scene, data)
    ids = [cols[int(t)] for t in mv.trackID if int(t) in cols]
    code = load_inst_code(scene, frame, data)
    return np.isin(code, ids) if ids else np.zeros(code.shape, bool)

def textureless_mask(rgb, win=9, thr=6.0):
    """True where the local average image gradient is small (flat areas)."""
    g = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (3, 3), 0)
    mag = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) ** 2 +
                  cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) ** 2)
    return cv2.blur(mag, (win, win)) < thr

def valid_depth_mask(depth_m):
    """Exclude pixels with no depth (sky is stored as 655.35 m)."""
    return (depth_m > 0) & (depth_m < 655.0)
