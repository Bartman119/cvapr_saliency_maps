#!/usr/bin/env python3
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

sys.path.append('../src')
from bilinear_inter_gray import bilinear_inter_gray

# def bilinear_inter_gray(img, a, b):
#     h, w = img.shape
#     out_h = int(h * a)
#     out_w = int(w * b)

#     xs, ys = np.meshgrid(range(out_w), range(out_h))  # output image index

#     _xs = np.floor(xs / b).astype(int)  # original x
#     _ys = np.floor(ys / a).astype(int)  # original y

#     dx = xs / b - _xs
#     dy = ys / a - _ys

#     _xs1p = np.minimum(_xs + 1, w - 1)
#     _ys1p = np.minimum(_ys + 1, h - 1)

#     out = (1 - dx) * (1 - dy) * img[_ys, _xs] + \
#             dx * (1 - dy) * img[_ys, _xs1p] + \
#             (1 - dx) * dy * img[_ys1p, _xs] + \
#             dx * dy * img[_ys1p, _xs1p]

#     return np.clip(out, 0, 255).astype(np.uint8)


def main():
    img_orig = io.imread('../images/zou_nantes_512x512.jpg')
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)

    sal = np.zeros_like(img_gray, dtype=np.float32)

    pyramid = [img_gray.astype(np.float32)]

    for i in range(1, 6):
        img_resized = bilinear_inter_gray(
            img_gray, a=1. / 2 ** i, b=1. / 2 ** i)
        img_resized = bilinear_inter_gray(img_resized, a=2 ** i, b=2 ** i)
        pyramid.append(img_resized.astype(np.float32))

    pyramid_n = len(pyramid)

    for i in range(pyramid_n):
        for j in range(pyramid_n):
            if i == j:
                continue
            sal += np.abs(pyramid[i] - pyramid[j])

    sal /= sal.max()
    sal *= 255
    sal = sal.astype(np.uint8)

    fig, ax = plt.subplots(1, 2, figsize=(6, 4))
    ax[0].set_title("input")
    ax[0].imshow(img_orig)
    ax[1].set_title("saliency")
    ax[1].imshow(sal, cmap="gray")

    # plt.savefig('../results/saliency_map.png')
    plt.show()


if __name__ == '__main__':
    main()
