import numpy as np
import matplotlib.pyplot as plt
import cv2

def polygon_to_mask(polygon, resolution_wh):
    """Generate a mask from a polygon.

    Args:
        polygon (np.ndarray): The polygon for which the mask should be generated,
            given as a list of vertices.
        resolution_wh (Tuple[int, int]): The width and height of the desired resolution.

    Returns:
        np.ndarray: The generated 2D mask, where the polygon is marked with
            `1`'s and the rest is filled with `0`'s.
    """
    width, height = resolution_wh
    mask = np.zeros((height, width))

    cv2.fillPoly(mask, [polygon], color=1)
    return mask

a = polygon_to_mask(np.array([[266, 191],
       [265, 192],
       [264, 192],
       [261, 203],
       [261, 204],
       [260, 205],
       [260, 206],
       [258, 207],
       [258, 208],
       [256, 210],
       [256, 211],
       [253, 213],
       [253, 214],
       [251, 215],
       [251, 216],
       [250, 217],
       [250, 218],
       [248, 219],
       [248, 220],
       [247, 221],
       [247, 222],
       [245, 223],
       [245, 224],
       [244, 225],
       [244, 229],
       [243, 230],
       [243, 234],
       [245, 235],
       [245, 236],
       [246, 237],
       [246, 238],
       [252, 245],
       [252, 246],
       [253, 247],
       [253, 248],
       [254, 249],
       [254, 250],
       [256, 252],
       [261, 252],
       [263, 250],
       [264, 250],
       [265, 249],
       [266, 249],
       [267, 248],
       [268, 248],
       [269, 247],
       [270, 247],
       [272, 245],
       [273, 245],
       [274, 244],
       [275, 244],
       [277, 242],
       [278, 242],
       [280, 240],
       [281, 240],
       [285, 236],
       [285, 235],
       [286, 234],
       [286, 230],
       [285, 229],
       [286, 228],
       [286, 226],
       [287, 225],
       [287, 224],
       [286, 223],
       [286, 222],
       [285, 221],
       [285, 220],
       [276, 211],
       [276, 210],
       [275, 209],
       [275, 207],
       [276, 206],
       [276, 205],
       [275, 204],
       [275, 202],
       [274, 201],
       [274, 200],
       [273, 199],
       [273, 198],
       [272, 197],
       [271, 197],
       [270, 196],
       [268, 196],
       [267, 195]]
), (640, 480))

plt.imshow(a)
plt.show()
