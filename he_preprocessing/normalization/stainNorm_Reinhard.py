"""
Normalize a patch stain to the target image using the method of:

E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
"""

from __future__ import division
import cv2 as cv
import numpy as np
import he_preprocessing.normalization.stain_utils as ut
from he_preprocessing.constants import bcolors


def lab_split(I):
    """
    Convert from RGB uint8 to LAB and split into channels
    :param I: uint8
    :return:
    """
    I = cv.cvtColor(I, cv.COLOR_RGB2LAB)
    I = I.astype(np.float32)
    I1, I2, I3 = cv.split(I)
    I1 /= 2.55
    I2 -= 128.0
    I3 -= 128.0
    return I1, I2, I3


def merge_back(I1, I2, I3):
    """
    Take seperate LAB channels and merge back to give RGB uint8
    :param I1:
    :param I2:
    :param I3:
    :return:
    """
    I1 *= 2.55
    I2 += 128.0
    I3 += 128.0
    I = np.clip(cv.merge((I1, I2, I3)), 0, 255).astype(np.uint8)
    return cv.cvtColor(I, cv.COLOR_LAB2RGB)


def get_mean_std(I):
    """
    Get mean and standard deviation of each channel
    :param I: uint8
    :return:
    """
    I1, I2, I3 = lab_split(I)
    m1, sd1 = cv.meanStdDev(I1)
    m2, sd2 = cv.meanStdDev(I2)
    m3, sd3 = cv.meanStdDev(I3)
    means = m1, m2, m3
    stds = sd1, sd2, sd3
    return means, stds


class Normalizer(object):
    """
    A stain normalization object
    """

    def __init__(self, standardize_brightness=True):
        self.target_means = None
        self.target_stds = None
        self.standardize_brightness = standardize_brightness

    def fit(self, target):
        if self.standardize_brightness:
            target, _ = ut.standardize_brightness(target)
        means, stds = get_mean_std(target)
        self.target_means = means
        self.target_stds = stds

    def get_99_percentile_saturation_vector(self):
        raise NotImplementedError

    def transform(self, I):
        if self.standardize_brightness:
            I, _ = ut.standardize_brightness(I)
        I_copy = np.copy(I)
        try:
            I1, I2, I3 = lab_split(I)
            means, stds = get_mean_std(I)
            norm1 = (
                (I1 - means[0]) * (self.target_stds[0] / stds[0])
            ) + self.target_means[0]
            norm2 = (
                (I2 - means[1]) * (self.target_stds[1] / stds[1])
            ) + self.target_means[1]
            norm3 = (
                (I3 - means[2]) * (self.target_stds[2] / stds[2])
            ) + self.target_means[2]
            return merge_back(norm1, norm2, norm3)
        except Exception as e:
            print(f"{bcolors.WARNING}Error: {e}. Returning input image.{bcolors.ENDC}")
            return I_copy.astype(np.uint8)