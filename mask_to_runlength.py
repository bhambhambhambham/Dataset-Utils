import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Any, Dict, Generator, ItemsView, List, Tuple

def mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def polygon_to_mask(polygon, width=1280, height=720):
    """
    Convert a polygon to a binary mask.
    
    Args:
        polygon (list of lists of int): List of polygon points.
        width (int): Width of the mask.
        height (int): Height of the mask.
    
    Returns:
        numpy array: Binary mask.
    """
    masks = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(masks, [polygon], color=1)
    return masks

def write_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

json_data = read_json_file('/home/bham/Desktop/valve-6d-poseestimation/data_storage/Dataset/datasets/TNS_valve_dataset_kinect/data/49/0-meta.json')
detection_ism = []

for key in json_data.keys():
    if(not json_data[key]['found']):
        continue
    else:
        data = json_data[key]['segmentation']
        mask = polygon_to_mask(data)

        tensor_mask = torch.tensor(mask, dtype=torch.uint8).unsqueeze(0)
        rle_mask = mask_to_rle_pytorch(tensor_mask)
        print(rle_mask)
        bbox = json_data[key]['bbox_2D_tight']
        print(bbox)

        detection_ism.append({
        "scene_id": 0,
        "image_id": 0,
        "category_id": 1,
        "bbox": bbox,
        "score": 0.3672521114349365,  # Example score, replace with actual if available
        "time": 0.0,  # Example time, replace with actual if available
        "segmentation": rle_mask[0]
        })
write_json_file('/home/bham/Desktop/6/1/detection_ism.json', detection_ism)
