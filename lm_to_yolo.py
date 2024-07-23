import shutil
import cv2
import matplotlib.pyplot as plt

object_list = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14 ,15]
continue_name_train = 0
continue_name_valid = 0

def mask_to_polygons(mask):
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

    return ' '.join([str(coord) for coord in polygons_2[0]])

for i, obj in enumerate(object_list):
    train_txt = open('../DenseFusion/datasets/linemod/Linemod_preprocessed/data/{0}/train.txt'.format('%02d' % obj))
    test_txt = open('../DenseFusion/datasets/linemod/Linemod_preprocessed/data/{0}/test.txt'.format('%02d' % obj))

    train_comp = train_txt.readlines()
    valid_comp = test_txt.readlines()

    src = '/home/bham/Desktop/DenseFusion/datasets/linemod/Linemod_preprocessed/data/{0}'.format('%02d' % obj)
    for ind, line in enumerate(train_comp):
        line = int(line.strip())

        images_name = str(continue_name_train+ind+1)+'.png'
        labels_name = str(continue_name_train+ind+1)+'.txt'

        img_src = src + '/rgb/{0}.png'.format('%04d' % line)
        shutil.copy(img_src, '/home/bham/Desktop/Dataset_prep/Datasets/images/train/'+images_name)

        mask_src = src + '/mask/{0}.png'.format('%04d' % line)
        mask = cv2.imread(mask_src, cv2.IMREAD_GRAYSCALE)

        message = str(i)+' '+mask_to_polygons(mask)
        with open('/home/bham/Desktop/Dataset_prep/Datasets/labels/train/'+labels_name, 'w') as f:
            f.write(message)
            f.close()

        last = ind + 1

    threshold = round(len(valid_comp)*0.8)
    for ind_2, line_2 in enumerate(valid_comp):
        line_2 = int(line_2.strip())

        images_name_2 = str(continue_name_valid+ind_2+1)+'.png'
        labels_name_2 = str(continue_name_valid+ind_2+1)+'.txt'

        img_src_2 = src + '/rgb/{0}.png'.format('%04d' % line_2)

        mask_src_2 = src + '/mask/{0}.png'.format('%04d' % line_2)
        mask_2 = cv2.imread(mask_src_2, cv2.IMREAD_GRAYSCALE)
        message = str(i)+' '+mask_to_polygons(mask_2)

        if ind_2 < threshold:
            shutil.copy(img_src_2, '/home/bham/Desktop/Dataset_prep/Datasets/images/train/a'+images_name_2)
            with open('/home/bham/Desktop/Dataset_prep/Datasets/labels/train/a'+labels_name_2, 'w') as f:
                f.write(message)
                f.close()
        else:
            shutil.copy(img_src_2, '/home/bham/Desktop/Dataset_prep/Datasets/images/val/'+images_name_2)
            with open('/home/bham/Desktop/Dataset_prep/Datasets/labels/val/'+labels_name_2, 'w') as f:
                f.write(message)
                f.close()

        last_2 = ind_2 + 1

    continue_name_train += last
    continue_name_valid += last_2
