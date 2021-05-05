import numpy as np
import numpy.typing as npt
import cv2

from path import Path

DATA_PATH = '../data/iam/ascii/words.txt'
IMG_SIZE = (32, 128) # h, w


def parse_line(line: str) -> tuple(Path, str):
    """
    Returns image path and word transcription from words.txt file of IAM dataset.

    Args:
        line: non comment line from IAM dataset's words.txt file.
    """

    split_line = line.strip('\n').split()
    assert len(split_line) >= 9
    
    word_id = split_line[0]
    word = ' '.join(split_line[8:])
    
    split_id = word_id.split('-')
    
    im_path = Path(f"../data/iam/words/{split_id[0]}/{split_id[0]}-{split_id[1]}/{word_id}.png")
    assert im_path.exists()
    
    return im_path, word


def resize_image(img: npt.ArrayLike, target_size: tuple(int, int)) -> npt.ArrayLike:
    """
    Resizes image 'img' to 'target_size'.

    Args:
        img: grayscale image with shape (height, width)
        target_size: size to convert 'img' to
    """

    ht, wt = target_size
    h, w = img.shape
    # scaling coefficient
    sc = min(wt / w, ht / h)
    tx = (wt - w * sc) / 2
    ty = (ht - h * sc) / 2
    
    # M = [[x_scale, x_shear, X_up_left], 
    #      [y_shear, y_scale, Y_up_left]]
    M = np.float32([[sc, 0, tx], [0, sc, ty]])
    img = cv2.warpAffine(img, M, dsize=(wt, ht), borderValue=0)

    return img / 255.
    

def iam_data_generator() -> tuple(npt.ArrayLike, str):
    """Yields image and word written on it."""
    
    with open(DATA_PATH) as iam:
        for line in iam:
            if not line.startswith('#'):
                im_path, word = parse_line(line)
                image = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
                image = resize_image(image, IMG_SIZE)
                yield image, word
            