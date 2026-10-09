import random
import numpy as np
overlap_size = 30

def remove_zero_padding(arr):

    left_bounding = 0
    right_bounding = 0

    t = 0
    for i in range(arr.shape[0]):
        if t == 1:
            break

        for j in range(arr.shape[1]):
            if not arr[i][j] == 0:
                left_bounding = i
                t = 1
                break

    t = 0
    for i in reversed(range(arr.shape[0])):
        if t == 1:
            break

        for j in range(arr.shape[1]):
            if not arr[i][j] == 0:
                right_bounding = i
                t = 1
                break

    return arr[:, left_bounding:right_bounding]

def concat(a, b, overlap=True, intersection_scalar=0.2):

    assert a.shape[0] == b.shape[0]

    if overlap is False:
        return np.concatenate((a, b), axis=1)

    sequence_length = a.shape[1] + b.shape[1]
    intersection_size = int(intersection_scalar * min(a.shape[1], b.shape[1]))

    im = np.zeros((a.shape[0], sequence_length - intersection_size))

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if not a[i][j] == 0:
                im[i][j] = a[i][j]

    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if not b[i][j] == 0:
                im[i][j + (a.shape[1] - intersection_size)] = b[i][j]

    return im

def merge(list_file, overlap_prob=True):

    im = np.zeros((28, 1))

    for (i, arr) in enumerate(list_file):
        arr = remove_zero_padding(arr)

        ins = 0
        ovp = False

        if overlap_prob is True:
            t = random.randint(1, overlap_size)
            ins = float(t / 100)

        if overlap_prob is True:
            ovp = random.choice([True, False])

        im = concat(im, arr, ovp, intersection_scalar=ins)

    return im
