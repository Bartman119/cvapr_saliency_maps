#!/usr/bin/env python3
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

sys.path.append('../src')
#from bilinear_inter_gray import bilinear_inter_gray


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

# def bilinear_interpolate(im, a, b):
#     x = np.asarray(a)
#     y = np.asarray(b)

#     x0 = np.floor(x).astype(int)
#     x1 = x0 + 1
#     y0 = np.floor(y).astype(int)
#     y1 = y0 + 1

#     x0 = np.clip(x0, 0, im.shape[1]-1);
#     x1 = np.clip(x1, 0, im.shape[1]-1);
#     y0 = np.clip(y0, 0, im.shape[0]-1);
#     y1 = np.clip(y1, 0, im.shape[0]-1);

#     Ia = im[ y0, x0 ]
#     Ib = im[ y1, x0 ]
#     Ic = im[ y0, x1 ]
#     Id = im[ y1, x1 ]

#     wa = (x1-x) * (y1-y)
#     wb = (x1-x) * (y-y0)
#     wc = (x-x0) * (y1-y)
#     wd = (x-x0) * (y-y0)

#     return wa*Ia + wb*Ib + wc*Ic + wd*Id


def main():
    img_orig = io.imread("C:/Users/pauli/PycharmProjects/ProjektCVAPR/images/dog.jpg")
    #img_orig = io.imread("../images/dog.jpg")
    #img_orig = io.imread("../images/zou_nantes_512x512.jpg")
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)

    pyramid = [img_gray]

    fig, ax = plt.subplots(1, 5, figsize=(10, 4))
    for i in range(1, 7):
        img_resized = bilinear_inter_gray(
            img_gray, a=1. / 2 ** i, b=1. / 2 ** i)
        pyramid.append(img_resized)
        ax[i - 1].set_title(f"1 / {2 ** i}")
        ax[i - 1].imshow(img_resized, cmap='gray')

    #plt.savefig('../results/gaussian_pyramid.png')
    plt.show()


if __name__ == '__main__':
    main()
