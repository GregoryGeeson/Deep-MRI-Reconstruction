from __future__ import print_function, division
import numpy as np
from scipy.io import loadmat
from os.path import join
from utils import mymath
# BEGIN-MPL
import matplotlib.pyplot as plt
# END-MPL


def show_slices(arr, skip=1):
    # BEGIN-MPL
    if len(arr.shape) >= 3:
        for i in range(0, arr.shape[0], skip):
            plt.imshow(arr[i], cmap='gray')
            plt.show()
    else:
        plt.imshow(arr, cmap='gray')
        plt.show()
    # END-MPL


def process_kspace(arr):
    # Sum up the raw data
    arr = np.sum(arr, axis=0)
    mag = np.absolute(arr)
    # Normalise and enhance contrast
    p = 0.1 * np.ones(mag.shape[1:])
    if len(mag.shape) >= 3:
        for i in range(mag.shape[0]):
            mag[i] = np.power(1 / np.amax(mag[i]) * mag[i], p)
    else:
        mag = 1 / np.amax(mag) * mag
    arr_im = mymath.ifft2c(arr)
    mag_im = np.absolute(arr_im)
    print("Magnitude dtype: {}".format(mag.dtype))
    return mag, mag_im


def imshow_compare(img1, img2):
    # BEGIN-MPL
    plt.subplot(121)
    plt.imshow(img1, cmap='gray')
    plt.subplot(122)
    plt.imshow(img2, cmap='gray')
    plt.show()
    # END-MPL


def training_allocation(data, allocation, method='random'):
    t, v, s = allocation
    if method == 'random':
        np.random.shuffle(data)
    return data[:t], data[t:t + v], data[t + v:]


if __name__ == '__main__':
    project_root = '/home/ben/projects/honours/dccnn/Deep-MRI-Reconstruction'
    # filename = join(project_root, 'data', 'cardiac.mat')
    # filename = join(project_root, 'data', 'lustig_knee_p2.mat')
    filename_raw = join(project_root, '..', '..', 'datasets', 'nyu_ml_knee', '1',
                        'rawdata15.mat')  # NYU knee raw k-space
    filename_img = join(project_root, '..', '..', 'datasets', 'nyu_ml_knee', '1',
                        'espirit15.mat')  # NYU knee images

    mr_data_raw = loadmat(filename_raw)
    mr_data_img = loadmat(filename_img)

    # mr_data = mr_data['xn']  # Lustig knee
    # mr_data = loadmat(filename)['seq']  # Schlemper cardiac
    mr_data_raw = mr_data_raw['rawdata']  # NYU knee rawdata
    mr_data_img = mr_data_img['reference']  # NYU knee espirit
    print("Raw data shape: {}".format(mr_data_img.shape))
    print("Raw data type: {}".format(mr_data_img.dtype))

    # nx, ny, nz, nc = mr_data.shape
    mr_data_raw = np.transpose(mr_data_raw, (2, 0, 1))  # .reshape((-1, ny, nz))

    mag, mag_im = process_kspace(mr_data_raw)
    imshow_compare(np.absolute(mr_data_img), mag_im)

    # train, validate, test = training_allocation(mr_data, [160, 40, 56])

    sk = 1
    # show_slices(train, skip=sk)
    # show_slices(validate, skip=sk)
    # show_slices(test, skip=sk)
