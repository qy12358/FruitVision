import cv2
import numpy as np


class ImagePreprocessor:

    def __init__(self, resize=(224, 224)):
        self.resize = resize

    def resize_image(self, image):
        return cv2.resize(image, self.resize)

    def bgr_to_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def gaussian_filter(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def histogram_equalization(self, hsv_image):
        h, s, v = cv2.split(hsv_image)
        v_equalized = cv2.equalizeHist(v)
        equalized = cv2.merge((h, s, v_equalized))
        return equalized

    def preprocess(self, image):
        resized = self.resize_image(image)
        hsv = self.bgr_to_hsv(resized)
        h, s, v = cv2.split(hsv)
        gaussian = self.gaussian_filter(hsv)
        equalized = self.histogram_equalization(gaussian)

        return {

            # Images
            "resized": resized,
            "hsv": hsv,
            "hue": h,
            "saturation": s,
            "value": v,
            "gaussian": gaussian,
            "equalized": equalized,

            # Metadata
            "original_size": image.shape,
            "resized_size": resized.shape,
            "kernel": (5, 5)
        }