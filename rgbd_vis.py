import open3d as o3d
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

rgb_path = 'valve1/rgb.png'
depth_path = 'valve1/depth.png'

rgb = Image.open(rgb_path)
depth = Image.open(depth_path)

print(np.array(rgb).shape)
print(np.array(depth).shape)

# plt.imshow(rgb)
# plt.show()




camera_intrinsic = [607.060302734375, 0.0, 0.0, 0.0, 607.1031494140625, 0.0, 639.758056640625, 363.29052734375, 1.0]

o3d_rgb = o3d.geometry.Image(np.array(Image.open(rgb_path)))
o3d_depth = o3d.geometry.Image(np.array(Image.open(depth_path)))

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

o3d.visualization.draw_geometries([pcd])