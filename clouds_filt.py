import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image

K = np.array([[607.060302734375, 0.0, 639.758056640625],
              [0.0, 607.1031494140625, 363.29052734375],
              [0.0, 0.0, 1.0]])

def depth_to_point_cloud(depth_img, mask, K):
    h, w = depth_img.shape
    i, j = np.indices((h, w))
    valid = (mask > 0) & (depth_img > 0)

    # Depth image to 3D point cloud
    z = depth_img[valid]
    x = (j[valid] - K[0, 2]) * z / K[0, 0]
    y = (i[valid] - K[1, 2]) * z / K[1, 1]
    
    points = np.stack((x, y, z), axis=-1) / 1000
    return points

def visualize_point_cloud(points, scene):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color([1, 0, 0])
    point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([scene, point_cloud])

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# rle: Dict[str, Any]
def rle_to_mask(rle) -> np.ndarray:
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()

file_path = 'valve2/detection_ism.json'
json_data = read_json_file(file_path)

image = cv2.imread('valve2/rgb.png')
rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
depth_img = np.array(Image.open('valve2/depth.png'))

camera_intrinsic = [607.060302734375, 0.0, 0.0, 0.0, 607.1031494140625, 0.0, 639.758056640625, 363.29052734375, 1.0]

o3d_rgb = o3d.geometry.Image(np.array(Image.open('valve2/rgb.png')))
o3d_depth = o3d.geometry.Image(np.array(Image.open('valve2/depth.png')))

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_depth)

fx = camera_intrinsic[0]
fy = camera_intrinsic[4]
cx = camera_intrinsic[6]
cy = camera_intrinsic[7]

intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, fx, fy, cx, cy)
camera_intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
intrinsic.intrinsic_matrix = camera_intrinsic_matrix

cam = o3d.camera.PinholeCameraParameters()
cam.intrinsic = intrinsic
cam.extrinsic = np.array([[1., 0., 0., 0.], [0.,1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
rgbd_image, cam.intrinsic, cam.extrinsic)
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# data = json_data[3]
for data in json_data:
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bbox = data['bbox']
    score = data['score']
    rle = data['segmentation']
    mask_img = rle_to_mask(rle)

        # Optional: Uncomment to display bounding boxes and masks using plt
    x, y, w, h = bbox
    cv2.rectangle(rgb_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    score_text = f"{score:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 255, 0)  # Green
    font_thickness = 1
    text_size, _ = cv2.getTextSize(score_text, font, font_scale, font_thickness)
    text_x = x + w - text_size[0]
    text_y = y - 10 if y - 10 > 10 else y + 10 + text_size[1]
    cv2.putText(rgb_img, score_text, (text_x, text_y), font, font_scale, font_color, font_thickness)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(rgb_img)
    axes[0].axis('off')
    axes[1].imshow(mask_img, cmap='gray')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    mask = mask_img > 0
    points = depth_to_point_cloud(depth_img, mask, K)
    visualize_point_cloud(points, pcd)
    
    # Optional: Uncomment to display image with bounding boxes
    # cv2.imshow('Image with Bounding Boxes', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
