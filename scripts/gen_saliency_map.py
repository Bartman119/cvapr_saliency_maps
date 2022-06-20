#!/usr/bin/env python3
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import pyautogui


sys.path.append('../src')
#from bilinear_inter_gray import bilinear_inter_gray

coords = []
user_coords = []
filename = ''

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

def generate_sailency(file_path):

    img_orig1 = io.imread(file_path)
    img_orig1 = cv2.resize(img_orig1, [512,512])
    img_gray1 = cv2.cvtColor(img_orig1, cv2.COLOR_RGB2GRAY)

    def onclick(event):
        global user_coords, filename
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print('x = %d, y = %d' % (
            ix, iy))

        global coords
        coords.append((ix, iy))
        user_coords.append((sal[int(iy)][int(ix)]))
        print(len(coords))

        if len(coords) == 3:
            ax[1].axis('on')
            ax[2].axis('on')
            ax[1].set_title("generated saliency")
            ax[1].imshow(sal, cmap="jet")
            ax[2].set_title("saliency from database")
            filename = filename[:-4]
            filename = filename + "_SaliencyMap.jpg"
            database_sailency = io.imread(filename)
            ax[2].imshow(database_sailency, cmap="jet")
            plt.show()

        if len(coords) == 4:
            user_mean = np.mean(user_coords)
            user_mean = int(user_mean * 100 / 255)
            user_min = min(user_coords)
            user_min = int(user_min * 100 / 255)
            user_max = max(user_coords)
            user_max = int(user_max * 100 / 255)
            text = str('Your results:\nMean: ' + str(user_mean) + '%\nMaximum: ' + str(user_max) + '%\nMinimum:' + str(
                user_min) + '%')
            pyautogui.alert(text, "End of the attempt")
            fig.canvas.mpl_disconnect(cid)
            plt.close()

        return

    # TODO: create gaussian maps and saliency for each image
    sal = np.zeros_like(img_gray1, dtype=np.float32)  # output saliency map generated from gaussian pyramid

    pyramid = [img_gray1.astype(np.float32)]

    for i in range(1, 10):
        img_resized = bilinear_inter_gray(
            img_gray1, a=1. / 2 ** i, b=1. / 2 ** i)
        img_resized = bilinear_inter_gray(img_resized, a=2 ** i, b=2 ** i)
        pyramid.append(img_resized.astype(np.float32))

    pyramid_n = len(pyramid)

    for i in range(pyramid_n):  # get the differences between each layer of gaussian pyramid
        for j in range(pyramid_n):
            if i == j:
                continue
            sal += np.abs(pyramid[i] - pyramid[j])

    sal /= sal.max()  # normalize image values
    sal *= 255
    sal = sal.astype(np.uint8)

    fig, ax = plt.subplots(1, 3, figsize=(6, 4))
    ax[0].set_title("input")
    ax[0].imshow(img_orig1)


    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    ax[1].axis('off')
    ax[2].axis('off')
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    print(coords)

def main():
    global filename, coords, user_coords
    for i in range(1, 192, 2):
        coords = []
        user_coords = []
        if i < 10:
            filename = "C:/Users/pauli/OneDrive/Pulpit/CVAPR/00"+str(i)+".jpg"
        elif i < 100:
            filename = "C:/Users/pauli/OneDrive/Pulpit/CVAPR/0" + str(i) + ".jpg"
        else:
            filename = "C:/Users/pauli/OneDrive/Pulpit/CVAPR/" + str(i) + ".jpg"

        generate_sailency(filename)

    #TODO: load 3 random images from dataset instead of 1



    # img_orig2 = io.imread("C:/Users/pauli/PycharmProjects/ProjektCVAPR/images/dog2.jpg")
    # img_orig2 = cv2.resize(img_orig2, [512,512])
    # img_gray2 = cv2.cvtColor(img_orig2, cv2.COLOR_RGB2GRAY)
    #
    # img_orig3 = io.imread("C:/Users/pauli/PycharmProjects/ProjektCVAPR/images/zou_nantes_512x512.jpg")
    # img_orig3 = cv2.resize(img_orig3, [512,512])
    # img_gray3 = cv2.cvtColor(img_orig3, cv2.COLOR_RGB2GRAY)

    #TODO: show 3 images, get coordinates of user's clicks
    #fig, ax = plt.subplots(1, 2, figsize=(6, 4))
    #ax[0].set_title("input")
    #ax[0].imshow(img_orig)

    #cid = fig.canvas.mpl_connect('button_press_event', onclick)


    #TODO: check sal value for each of the clicked coordinates, 
    # show original images and their respective saliency maps
    # if greater than value (>200 maybe) print information that region of interest was successfully marked 

    


if __name__ == '__main__':
    main()
