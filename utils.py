from pycocotools import mask as maskUtils
import numpy as np
import matplotlib.pyplot as plt

def polygons_to_mask(segmentations, height, width):
    rles = maskUtils.frPyObjects(segmentations, height, width)
    rle = maskUtils.merge(rles)
    mask = maskUtils.decode(rle)
    return mask
