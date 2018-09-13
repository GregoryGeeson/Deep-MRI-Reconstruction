from __future__ import print_function, division
import numpy as np
from scipy.io import loadmat
from os.path import join
# BEGIN-MPL
import matplotlib.pyplot as plt
# END-MPL


def show_slices(arr, skip=1):
    print("Shape: {}".format(arr.shape))
    print("Original dtype: {}".format(arr.dtype))
    mag = np.absolute(arr)
    print("Magnitude dtype: {}".format(mag.dtype))

    # BEGIN-MPL
    for i in range(0, mag.shape[0], skip):
        plt.imshow(mag[i], cmap='gray')
        plt.show()
    # END-MPL


def training_allocation(data, allocation, method='random'):
    t, v, s = allocation
    if method == 'random':
        np.random.shuffle(data)
    return data[:t], data[t:t+v], data[t+v:]


if __name__ == '__main__':
    project_root = '.'
    mr_data = loadmat(join(project_root, './data/lustig_knee_p2.mat'))['xn']
    nx, ny, nz, nc = mr_data.shape
    print("Raw data shape: {}".format(mr_data.shape))
    print("Raw data type: {}".format(mr_data.dtype))
    mr_data = np.transpose(mr_data, (3, 0, 1, 2)).reshape((-1, ny, nz))
    print("Reshaped data shape: {}".format(mr_data.shape))
    print("Reshaped data type: {}".format(mr_data.dtype))
    train, validate, test = training_allocation(mr_data, [160, 40, 56])

    show_slices(train, skip=16)
    show_slices(validate, skip=16)
    show_slices(test, skip=16)

