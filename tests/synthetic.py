"""
Procedurally generated RGB test scenes.  They are used ONLY to exercise the
code (smoke tests); they are not a dataset, are never used for training and no
accuracy figure is derived from them.
"""
import cv2
import numpy as np


def _noise(rng, h, w, scale):
    n = rng.standard_normal((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    return n / (n.std() + 1e-6)


def forest(seed=0, size=384, base=(38, 72, 38)):
    rng = np.random.default_rng(seed)
    h = w = size
    tex = 0.5 * _noise(rng, h, w, 3) + 0.35 * _noise(rng, h, w, 8) + 0.25 * _noise(rng, h, w, 24)
    img = np.array(base, np.float32)[None, None, :] * (1 + 0.22 * tex[..., None])
    img[..., 0] *= 1 + 0.05 * _noise(rng, h, w, 12)
    img += rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def water(seed=0, size=384, base=(22, 62, 112)):
    rng = np.random.default_rng(seed)
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = 0.10 * (xx / w - 0.5) + 0.06 * _noise(rng, h, w, 96)
    img = np.array(base, np.float32)[None, None, :] * (1 + grad[..., None])
    img += rng.normal(0, 2.0, img.shape) + 2.0 * _noise(rng, h, w, 6)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def agriculture(seed=0, size=384, palette=None):
    rng = np.random.default_rng(seed)
    palette = palette or [(96, 146, 62), (190, 168, 108), (150, 108, 72), (208, 196, 110),
                          (70, 120, 50), (170, 150, 90), (120, 160, 80)]
    img = np.zeros((size, size, 3), np.float32)

    def split(x0, y0, x1, y1, depth):
        if depth == 0 or (x1 - x0) < 60 or (y1 - y0) < 60:
            col = np.array(palette[rng.integers(len(palette))], np.float32) * rng.uniform(0.85, 1.12)
            reg = img[y0:y1, x0:x1]
            reg[:] = col
            rows = np.sin(np.arange(x1 - x0) * rng.uniform(0.6, 1.2)) if rng.random() < 0.5 else 0
            reg += (3.0 * rows)[..., None] if np.ndim(rows) else 0
            reg[:1] = reg[:1] * 0.8 + 40
            reg[:, :1] = reg[:, :1] * 0.8 + 40
            return
        if rng.random() < 0.5:
            c = int(rng.uniform(0.35, 0.65) * (x1 - x0)) + x0
            split(x0, y0, c, y1, depth - 1); split(c, y0, x1, y1, depth - 1)
        else:
            c = int(rng.uniform(0.35, 0.65) * (y1 - y0)) + y0
            split(x0, y0, x1, c, depth - 1); split(x0, c, x1, y1, depth - 1)

    split(0, 0, size, size, 4)
    img += rng.normal(0, 3.0, img.shape) + 3 * _noise(rng, size, size, 5)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def urban(seed=0, size=384):
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), 140, np.float32)               # asphalt / streets
    step = 32
    roofs = [(190, 190, 195), (170, 120, 100), (215, 210, 200), (120, 120, 125),
             (200, 170, 140), (95, 100, 110), (230, 230, 232)]
    for y in range(0, size, step):
        for x in range(0, size, step):
            for _ in range(2):
                bw, bh = rng.integers(8, 15), rng.integers(8, 15)
                bx, by = x + rng.integers(3, step - bw - 2), y + rng.integers(3, step - bh - 2)
                col = np.array(roofs[rng.integers(len(roofs))], np.float32) * rng.uniform(0.85, 1.1)
                img[by:by + bh, bx:bx + bw] = col
                img[by + bh:by + bh + 2, bx:bx + bw] *= 0.55          # shadow
            if rng.random() < 0.10:
                img[y + 4:y + step - 4, x + 4:x + step - 4] = (70, 110, 60)   # small park
    img += rng.normal(0, 3.0, img.shape)
    M = cv2.getRotationMatrix2D((size / 2, size / 2), float(rng.uniform(-15, 15)), 1.0)
    img = cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REFLECT)
    return np.clip(img, 0, 255).astype(np.uint8)


def river_scene(seed=0, size=384):
    """Mixed scene (forest + river) - used to check the app does not claim high confidence."""
    f, w = forest(seed, size), water(seed, size)
    yy, xx = np.mgrid[0:size, 0:size]
    band = np.abs(yy - (size // 2 + 60 * np.sin(xx / 45.0))) < 38
    out = f.copy(); out[band] = w[band]
    return out


def all_scenes():
    return {
        "forest_a": forest(1), "forest_b": forest(2, base=(55, 85, 45)),
        "forest_c": forest(3, base=(28, 55, 32)),
        "water_a": water(1), "water_b": water(2, base=(40, 92, 100)),
        "water_c": water(3, base=(25, 48, 58)),
        "agri_a": agriculture(1), "agri_b": agriculture(2), "agri_c": agriculture(3),
        "urban_a": urban(1), "urban_b": urban(2), "urban_c": urban(3),
    }
