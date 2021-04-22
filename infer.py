import os
import numpy as np
import cv2

from htr_models.m1 import Model


class Batch:
    """Batch containing images and ground truth texts"""
    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts


def preprocess(img, img_size):
    """
    Crop top left part of the image, put it into target 'img_size', transpose for TF 
    and normalize gray-values
    """

    img = img.astype(np.float)
    img = img[:40, :200]

    # center image
    wt, ht = img_size
    h, w = img.shape
    f = min(wt / w, ht / h)
    tx = (wt - w * f) / 2
    ty = (ht - h * f) / 2

    # map image into target image
    M = np.float32([[f, 0, tx], [0, f, ty]])
    target = np.ones(img_size[::-1]) * 255 / 2
    img = cv2.warpAffine(img, M, dsize=img_size, dst=target, borderMode=cv2.BORDER_TRANSPARENT)

    # transpose for TF
    img = cv2.transpose(img)

    # normalize values to be in range [-1, 1]
    img = img / 255 - 0.5
    return img

def infer(model, img_path, img_size=(128, 32)):
    """Recognize text in image provided by 'img_path', using given 'model'"""
    img = preprocess(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), img_size)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)

    return recognized, probability

def get_text_from_img(img_path, charlist_path, model_path):
    """Return recognized text from image"""
    model = Model(open(charlist_path).read(), decoderType=0, mustRestore=True, 
                  dump=False, modelDir=model_path)
    recognized, probability = infer(model, img_path)
    return recognized[0], probability
