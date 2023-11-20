from pathlib import Path

import numpy as np
from PIL import Image


def arr_from_img(path: Path):
    image = Image.open(path)
    data = image.getdata()
    return np.asarray([(pixel[0] / 255) for pixel in data], dtype=np.float32)
