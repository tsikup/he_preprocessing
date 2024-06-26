"""
Stain normalization inspired by method of:

A. Vahadane et al., ‘Structure-Preserving Color Normalization and Sparse Stain Separation for Histological Images’, IEEE Transactions on Medical Imaging, vol. 35, no. 8, pp. 1962–1971, Aug. 2016.

Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams or spams-bin
"""

from __future__ import division
import spams
import numpy as np
import he_preprocessing.normalization.stain_utils as ut
from he_preprocessing.constants import bcolors


def get_stain_matrix(I, threshold=0.8, lamda=0.1, numThreads=1):
    """
    Get 2x3 stain matrix. First row H and second row E
    :param I:
    :param threshold:
    :param lamda:
    :return:
    """
    mask = ut.notwhite_mask(I, thresh=threshold).reshape((-1,))
    OD = ut.RGB_to_OD(I).reshape((-1, 3))
    OD = OD[mask]
    dictionary = spams.trainDL(
        OD.T,
        K=2,
        lambda1=lamda,
        mode=2,
        modeD=0,
        posAlpha=True,
        posD=True,
        verbose=False,
        numThreads=numThreads,
    ).T
    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    dictionary = ut.normalize_rows(dictionary)
    return dictionary


###


class Normalizer(object):
    """
    A stain normalization object
    """

    def __init__(
        self, standardize_brightness=True, dataset_stain_matrix_reference=None
    ):
        self.stain_matrix_target = dataset_stain_matrix_reference
        self.standardize_brightness = standardize_brightness

    def fit(self, target):
        if self.standardize_brightness:
            target, _ = ut.standardize_brightness(target)
        self.stain_matrix_target = get_stain_matrix(target)

    def get_stain_vectors(self):
        return self.stain_matrix_target

    def target_stains(self):
        return ut.OD_to_RGB(self.stain_matrix_target)

    def transform(self, I, slide_stain_matrix_reference=None):
        if self.standardize_brightness:
            I, _ = ut.standardize_brightness(I)
        if slide_stain_matrix_reference is not None:
            stain_matrix_source = slide_stain_matrix_reference
        else:
            stain_matrix_source = get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)
        return (
            255
            * np.exp(
                -1
                * np.dot(source_concentrations, self.stain_matrix_target).reshape(
                    I.shape
                )
            )
        ).astype(np.uint8)

    def hematoxylin(self, I):
        if self.standardize_brightness:
            I, _ = ut.standardize_brightness(I)
        I_copy = np.copy(I)
        try:
            h, w, c = I.shape
            stain_matrix_source = get_stain_matrix(I)
            source_concentrations = ut.get_concentrations(I, stain_matrix_source)
            H = source_concentrations[:, 0].reshape(h, w)
            H = np.exp(-1 * H)
            return H
        except Exception as e:
            print(f"{bcolors.WARNING}Error: {e}. Returning input image.{bcolors.ENDC}")
            return I_copy.astype(np.uint8)
