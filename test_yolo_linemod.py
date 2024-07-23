from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

model = YOLO('best.pt')

img_path = '/home/bham/Desktop/DenseFusion/datasets/linemod/Linemod_preprocessed/data/08/rgb/0004.png'
rgb_image = cv2.imread(img_path)
results = model.predict(rgb_image, conf=0.7, retina_masks=True, show_labels=True)

img = Image.open(img_path)
plt.imshow(img)
plt.show()

for res in results:
    # print(res.masks.data)
    # print(res.boxes.data)
    # res.save_txt('a.txt')
    # print(res.names)
    mask_tensor = res.masks.data.squeeze(0).cpu()
    mask_img = (mask_tensor.numpy() * 255).astype(np.uint8)  # Convert to 8-bit format
    plt.imshow(mask_img)
    plt.show()

    # masked_image = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_img)
    # masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

    boxes = res.boxes.data
    object_id = int(boxes[0][5].cpu().numpy())
    print(object_id)
    # plt.imshow(masked_image)
    # plt.show()
