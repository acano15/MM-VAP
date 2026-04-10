# -*- coding: utf-8 -*-
from typing import Any

import numpy as np
import cv2


class CImageProcessing:
    """Image processing with cv2"""
    def __init__(self):
        pass

    @staticmethod
    def convert_to_grayscale(image: np.ndarray, give_three_channels: bool = True) -> np.ndarray:
        """
        Convert an RGB image to grayscale.

        Args:
            image (np.ndarray): The input RGB image as a NumPy array
            give_three_channels (bool): If True, the output grayscale image will have three channels

        Returns:
            np.ndarray: The converted grayscale image, either single-channel or three-channel.
        """
        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # If three channels are requested, convert the grayscale image to three channels
        if give_three_channels:
            grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

        return grayscale_image

    @staticmethod
    def crop_image_bbox(image: np.array(float), bbox_object: list) -> np.array(float):
        """
        Crop original image to the one within the bbox
        image dimension = [height, width, channels]

        Args:
            image: array image
            bbox_object: list of bounding box

        Returns:
            Cropped image.

        """
        x, y, width, height = bbox_object
        return image[y:y + height, x:x + width]

    @staticmethod
    def crop_image_bbox2(image: np.array(float), bbox_object: Any) -> np.array(float):
        """
        Crop original image to the one within the bbox

        Args:
            image: array image
            bbox_object: list of bounding box

        Returns:
            Cropped image.

        """
        x1, y1, x2, y2 = bbox_object
        height = y2 - y1
        width = x2 - x1
        return image[y1:y1 + height, x1:x1 + width]

    @staticmethod
    def image_resize_ratio(image: np.array(float), width: int, height: int,
                           inter: Any = None) -> np.array(float):
        """
        @author: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv

        Args:
            image: array image
            width: new width
            height: new height
            inter: Any:

        Returns:
            Resized image.

        """
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        if inter:
            interpolation = inter
        else:
            dif = h if h > w else w
            interpolation = cv2.INTER_AREA if dif > (width + height) // 2 else cv2.INTER_CUBIC

        resized = cv2.resize(image, dim, interpolation=interpolation)

        # return the resized image
        return resized

    @staticmethod
    def image_resize(image: np.array(float), width: int, height: int, inter: Any = None) -> \
        np.array(float):
        """
        @author: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv

        Args:
            image: array image
            width: new width
            height: new height
            inter: Any:

        Returns:
            Resized image.

        """
        h, w = image.shape[:2]
        c = image.shape[2] if len(image.shape) > 2 else 1

        if h == w:
            return cv2.resize(image, (width, height), cv2.INTER_CUBIC)

        dif = h if h > w else w
        if inter:
            interpolation = inter
        else:
            interpolation = cv2.INTER_AREA if dif > (width + height) // 2 else cv2.INTER_CUBIC

        x_pos = (dif - w) // 2
        y_pos = (dif - h) // 2

        if len(image.shape) == 2:
            mask = np.zeros((dif, dif), dtype=image.dtype)
            mask[y_pos:y_pos + h, x_pos:x_pos + w] = image[:h, :w]
        else:
            mask = np.zeros((dif, dif, c), dtype=image.dtype)
            mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = image[:h, :w, :]
        return cv2.resize(mask, (width, height), interpolation)

    def image_resize_rectangle(self, image: np.array(float), width: int, height: int,
                               inter: Any) -> np.array(float):
        """
        @author: https://stackoverflow.com/questions/44231209/resize-rectangular-image-to-square-keeping-ratio-and-fill-background-with-black

        Args:
            image: array image
            width: new width
            height: new height
            inter: Any:

        Returns:
            Resized image.

        """
        image = image.copy()
        s = max(image.shape[0:2])

        # Creating a dark square with NUMPY
        f = np.zeros((s, s, 3), np.uint8)

        # Getting the centering position
        ax, ay = (s - image.shape[1]) // 2, (s - image.shape[0]) // 2
        f[ay:image.shape[0] + ay, ax:ax + image.shape[1]] = image

        h, w = image.shape[:2]
        dif = h if h > w else w
        if inter:
            interpolation = inter
        else:
            interpolation = cv2.INTER_AREA if dif > (width + height) // 2 else cv2.INTER_CUBIC

        return self.image_resize_ratio(image, width, height, interpolation)
