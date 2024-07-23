import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image


def get_scene(rgb, depth, K):
    o3d_rgb = o3d.geometry.Image(rgb)
    o3d_depth = o3d.geometry.Image(depth)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_depth)

    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(1280, 720, fx, fy, cx, cy)
    camera_intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    intrinsic.intrinsic_matrix = camera_intrinsic_matrix

    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = intrinsic
    cam.extrinsic = np.array([[1., 0., 0., 0.], [0.,1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image, cam.intrinsic, cam.extrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd


def mask_to_cloud(depth_img, mask, K):
    # convert Mask and Depth to Clouds
    h, w = depth_img.shape
    i, j = np.indices((h, w))
    valid = (mask > 0) & (depth_img > 0)

    z = depth_img[valid]
    x = (j[valid] - K[0, 2]) * z / K[0, 0]
    y = (i[valid] - K[1, 2]) * z / K[1, 1]

    # Rescale based on the depth scale
    points = np.stack((x, y, z), axis=-1) / 1000

    # convert numpy points to clouds (Ready to be visualized)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color([0, 0, 1])
    point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return point_cloud

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

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

if __name__ == '__main__':
    K = np.array([[607.060302734375, 0.0, 639.758056640625],
              [0.0, 607.1031494140625, 363.29052734375],
              [0.0, 0.0, 1.0]])
    
    x = str(input("Object : "))
    y = str(input("Scene : "))
    
    Header = '/home/bham/Desktop/'+x+'/'
    Sample = y+'/'
    file_path = Header + Sample +'detection_pem.json'
    rgb_image_path = Header + Sample +'rgb.png'
    depth_image_path = Header + Sample +'depth.png'
    ply_path = Header + 'obj_000005.ply'
    # ply_path = '/home/bham/Desktop/valve-6d-poseestimation/data_storage/Dataset/datasets/TNS_valve_dataset_kinect/models/6_multi_turn_valve_3/points.ply'

    model_pcd = o3d.io.read_point_cloud(ply_path)
    model_points = np.asarray(model_pcd.points)
    json_data = read_json_file(file_path)

    rgb_img = np.array(Image.open(rgb_image_path))
    depth_img = np.array(Image.open(depth_image_path))

    # plt.imshow(rgb_img)
    # plt.show()

    scene = get_scene(rgb_img, depth_img, K)
    interested_clouds = o3d.geometry.PointCloud()
    predicted_clouds = o3d.geometry.PointCloud()

    num_mask = 0
    pem = False

    thresh = float(input("Confidence Threshold ([0, 1]) : "))

    print("Number of proposal : "+str(len(json_data)))
    for data in json_data:
        if data['score'] <= thresh:
            continue
        mask_img = rle_to_mask(data['segmentation'])
        interested_cloud = mask_to_cloud(depth_img, mask_img, K)

        if 'R' in data.keys():
            pem = True
            R = np.array(data['R'])
            t = np.array(data['t'])
            predicted_points = model_points.dot(R.T) + t

            predicted_cloud = o3d.geometry.PointCloud()
            predicted_cloud.points = o3d.utility.Vector3dVector(predicted_points)
            predicted_cloud.paint_uniform_color([1, 0, 0])
            predicted_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            predicted_clouds += predicted_cloud

        # o3d.visualization.draw_geometries([scene, predicted_cloud, interested_cloud])

        interested_clouds += interested_cloud

    if pem:
        o3d.visualization.draw_geometries([scene, interested_clouds, predicted_clouds])
    else:
        o3d.visualization.draw_geometries([scene, interested_clouds])

