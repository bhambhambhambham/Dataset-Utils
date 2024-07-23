import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def mask_to_polygons1(mask):
    height, width = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        if contour.shape[0] >= 3:
            normalized_contour = contour.astype(np.float32) / [width, height]
            flattened_contour = normalized_contour.reshape(-1).tolist()
            polygons.append(flattened_contour)
    
    return polygons[0]
#SV

def mask_to_polygons2(mask):
    H_2, W_2 = mask.shape
    contours_2, hierarchy_2 = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons_2 = []
    for cnt_2 in contours_2:
        if cv2.contourArea(cnt_2) > 20:
            polygon_2 = []
            for point_2 in cnt_2:
                x_2, y_2 = point_2[0]
                polygon_2.append(x_2 / W_2)
                polygon_2.append(y_2 / H_2)
            polygons_2.append(polygon_2)
    
    return polygons_2[0]

def unnormalize_coordinates(normalized_coords, width=640, height=480):
    unnormalized_coords = []
    for i in range(0, len(normalized_coords), 2):
        x = int(normalized_coords[i] * width)
        y = int(normalized_coords[i + 1] * height)
        unnormalized_coords.extend([x, y])
    return unnormalized_coords

def polygon_to_mask(polygon, width=640, height=480):
    masks = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(masks, [polygon], color=1)
    return masks

def visualize_masks(gt_mask, mask_1, mask_2):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gt_mask, cmap='gray')
    axes[0].set_title('Ground Truth Mask')
    axes[0].axis('off')
    
    axes[1].imshow(mask_1, cmap='gray')
    axes[1].set_title('Mask 1')
    axes[1].axis('off')
    
    axes[2].imshow(mask_2, cmap='gray')
    axes[2].set_title('Mask 2')
    axes[2].axis('off')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

def string_to_list(input_string):
    # Split the string by spaces and convert to a list
    result_list = input_string.split()
    
    # Convert the list items to float (if needed, otherwise remove this step)
    result_list = [float(item) for item in result_list]
    
    return result_list


# mask_gray = cv2.imread('/home/bham/Desktop/DenseFusion/datasets/linemod/Linemod_preprocessed/data/01/mask/0000.png', cv2.IMREAD_GRAYSCALE)
# # print(mask_to_polygons1(mask_gray))
# # print('---------------------')
# # print(mask_to_polygons2(mask_gray))
# # print(type(mask_to_polygons1(mask_gray)))
# a = mask_to_polygons1(mask_gray)
# b = mask_to_polygons2(mask_gray)

# # if a != b:
# a_unnorm = unnormalize_coordinates(a)
# b_unnorm = unnormalize_coordinates(b)

# # print(np.array(a_unnorm))
# # print('-------------------')
# # print(np.array(b_unnorm))

# mask_a = polygon_to_mask(np.array(a_unnorm))
# mask_b = polygon_to_mask(np.array(b_unnorm))

# visualize_masks(mask_gray, mask_a, mask_b)

# dir = os.listdir('/home/bham/Desktop/DenseFusion/datasets/linemod/Linemod_preprocessed/data/09/mask')

# for elem in dir:
#     mask_gray = cv2.imread('/home/bham/Desktop/DenseFusion/datasets/linemod/Linemod_preprocessed/data/09/mask/'+str(elem), cv2.IMREAD_GRAYSCALE)
#     a = mask_to_polygons1(mask_gray)
#     b = mask_to_polygons2(mask_gray)

#     a_unnorm = unnormalize_coordinates(a)
#     b_unnorm = unnormalize_coordinates(b)

#     mask_a = polygon_to_mask(np.array(a_unnorm))
#     mask_b = polygon_to_mask(np.array(b_unnorm))
#     # visualize_masks(mask_gray, mask_a, mask_b)
#     if a == b:
#         # print('True')
#         pass
#     else:
#         print(elem)
#         a_unnorm = unnormalize_coordinates(a)
#         b_unnorm = unnormalize_coordinates(b)

#         mask_a = polygon_to_mask(np.array(a_unnorm))
#         mask_b = polygon_to_mask(np.array(b_unnorm))
#         visualize_masks(mask_gray, mask_a, mask_b)


string = '0 0.4515625 0.7791666666666667 0.4390625 0.7777777777777778 0.4328125 0.7722222222222223 0.4328125 0.7625 0.415625 0.6805555555555556 0.4109375 0.6736111111111112 0.40390625 0.6472222222222223 0.40625 0.6291666666666667 0.4140625 0.6208333333333333 0.41875 0.6208333333333333 0.425 0.6291666666666667 0.43203125 0.6569444444444444 0.45546875 0.7597222222222222 0.45546875 0.7736111111111111 0.4515625 0.7791666666666667'
l = string_to_list(string)
l_np = np.array(l[1:])

l_unnorm = unnormalize_coordinates(l_np)
mask_l = polygon_to_mask(l_unnorm)
plt.imshow(mask_l, cmap = 'gray')
plt.show()