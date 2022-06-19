import numpy as np

def bilinear_inter_gray(img, a, b):
    h, w = img.shape
    out_h = int(h * a)
    out_w = int(w * b)

    xs, ys = np.meshgrid(range(out_w), range(out_h))  # output image index

    _xs = np.floor(xs / b).astype(int)  # original x
    _ys = np.floor(ys / a).astype(int)  # original y

    dx = xs / b - _xs
    dy = ys / a - _ys

    _xs1p = np.minimum(_xs + 1, w - 1)
    _ys1p = np.minimum(_ys + 1, h - 1)

    out = (1 - dx) * (1 - dy) * img[_ys, _xs] + \
            dx * (1 - dy) * img[_ys, _xs1p] + \
            (1 - dx) * dy * img[_ys1p, _xs] + \
            dx * dy * img[_ys1p, _xs1p]

    return np.clip(out, 0, 255).astype(np.uint8)
