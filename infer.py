import os
import numpy as np
import cv2

from htr_models.m1 import Model
from segmentation import get_lines_from_image, get_words_from_lines


class Batch:
    """Batch containing images and ground truth texts"""
    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts


def preprocess(img, img_size):
    """
    Put it into target 'img_size', transpose for TF and normalize gray-values
    """
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

def infer(model, img_path=None, img=None, img_size=(128, 32)):
    """Recognize text in image provided by 'img_path', using given 'model'"""
    if img_path is None and img is None:
        raise ValueError
    if img is None and img_path is not None:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = preprocess(img, img_size)
    batch = Batch(None, [img])
    (recognized, probability) = model.inferBatch(batch, True)

    return recognized, probability

def get_text_from_words(words, model):
    text = []
    errors = []
    for word in words:
        try:
            r, p = infer(model, img=word)
            text.append((r[0], p[0]))
        except ValueError:
            errors.append(word)

    return ' '.join(map(lambda t: t[0], text))

def get_text_from_img(img_path, charlist_path, model_path):
    """Return recognized text from image"""
    model = Model(open(charlist_path).read(), decoderType=0, mustRestore=True, 
                  dump=False, modelDir=model_path)

    img = cv2.imread(img_path)
    lines = get_lines_from_image(img)
    words = get_words_from_lines(lines)
    text = get_text_from_words(words, model)
    
    return text
