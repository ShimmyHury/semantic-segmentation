import os
import cv2
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def calc_class_weights(path_to_labels):
    ids = os.listdir(path_to_labels)
    masks_fps = [os.path.join(path_to_labels, image_id) for image_id in ids]
    mask_all = []
    for i in range(len(masks_fps)):
        mask = cv2.imread(masks_fps[i], 0)
        mask_all.append(mask.reshape(-1))

    mask_vec = np.array(mask_all).reshape(-1)
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(mask_vec), y=mask_vec)
    return weights


def one_hot_to_class(tragets):
    summation = tragets.sum(axis=1)
    classes = tragets.argmax(axis=1)
    classes[summation == 0] = -100
    return classes


def miou(targets, predictions, num_classes=12):
    iou_list = []
    present_iou_list = []
    present_wiou_list = []

    pred = predictions.flatten()
    label = targets.flatten()
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.sum() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).sum()
            union_now = pred_inds.sum() + target_inds.sum() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
            present_wiou_list.append(iou_now * (target_inds.sum() / label.size))
        iou_list.append(iou_now)
    return np.mean(present_iou_list), np.sum(present_wiou_list)
