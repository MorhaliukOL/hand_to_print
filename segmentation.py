import math
import numpy as np
import cv2

from scipy.signal import argrelmin



def smooth(x, window_len=11, window='hanning'):

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.") 
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'") 
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')
    return y

def normalize(img):
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    return img

def applySummFunctin(img):
    res = np.sum(img, axis = 0)    #  summ elements in columns
    return res

def createKernel(kernelSize, sigma, theta):
    "create anisotropic filter kernel according to given parameters"
    assert kernelSize % 2 # must be odd size
    halfSize = kernelSize // 2

    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize

            expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
            xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
            yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)

            kernel[i, j] = (xTerm + yTerm) * expTerm

    kernel = kernel / np.sum(kernel)
    return kernel

def crop_text_to_lines(text, blanks):
    x1 = 0
    y = 0
    lines = []
    for i, blank in enumerate(blanks):
        x2 = blank
        line = text[:, x1:x2]
        lines.append(line)
        x1 = blank
    return lines

def transpose_lines(lines):
    res = []
    for l in lines:
        line = np.transpose(l)
        res.append(line)
    return res

def get_lines_from_image(img, kernel_size=11, sigma=4, theta=2):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.transpose(img)
    img2 = cv2.filter2D(img, -1, createKernel(kernel_size, sigma, theta), 
                        borderType=cv2.BORDER_REPLICATE)
    img2 = normalize(img2)
    summ = applySummFunctin(img2)
    smoothed = smooth(summ, 35)
    mins = argrelmin(smoothed, order=2)
    arr_mins = np.array(mins)
    lines = crop_text_to_lines(img, arr_mins[0])
    lines = transpose_lines(lines) # why no use 'map()' ?
    return lines


def wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=0):
    """Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf
    Args:
        img: grayscale uint8 image of the text-line to be segmented.
        kernelSize: size of filter kernel, must be an odd integer.
        sigma: standard deviation of Gaussian function used for filter kernel.
        theta: approximated width/height ratio of words, filter function is distorted by this factor.
        minArea: ignore word candidates smaller than specified area.
    Returns:
        List of tuples. Each tuple contains the bounding box and the image of the segmented word.
    """
    kernel = createKernel(kernelSize, sigma, theta)
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    imgThres = 255 - imgThres

    if cv2.__version__.startswith('3.'):
        (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # cv2.RETR_EXTERNAL or cv2.RETR_LIST - ???
        (components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # append components to result
    res = []
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < minArea:
            continue
        # append bounding box and image of word to result list
        currBox = cv2.boundingRect(c)  # returns (x, y, w, h)
        (x, y, w, h) = currBox
        currImg = img[y:y + h, x:x + w]
        res.append((currBox, currImg))

    return sorted(res, key=lambda entry: entry[0][0])

def get_words_from_lines(lines):
    words = []
    for line in lines:
        for word in wordSegmentation(line):
            words.append(word[1])
    return words