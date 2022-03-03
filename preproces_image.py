import cv2
import numpy as np

image_size = (250, 250)


def prepare_image_for_network(image_path):
    img = cv2.imread(image_path)
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, image_size)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')
    img /= 255.

    return img
