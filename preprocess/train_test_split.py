import numpy as np
from . import pre_process_by_yield as pre


def train_test_split(data, label, label_unique, train=10, split_by_3d=False):
    assert data.ndim == 3
    assert label.ndim == 2

    label_pos = get_lables_pos(label, label_unique=label_unique)
    if isinstance(train, int):
        train_label_pos, test_label_pos = divide_labels_by_nums(
            label_pos, train)
    elif isinstance(train, float):
        train_label_pos, test_label_pos = divide_labels_by_ratio(
            label_pos, train)
    if split_by_3d is True:
        train, test = get_data_divided_by_3d(
            train_label_pos, test_label_pos, data)
    else:
        train, test = get_data_divided(train_label_pos, test_label_pos, data)

    return train, test


def get_lables_pos(label_array, label_unique=None):
    # get lable array for unique label
    # get number of unique label and it number
    if label_unique is None:
        label_unique = np.unique(label_array)
    else:
        label_unique = np.unique(label_unique)
    label_pos = {}
    # get a dictionary for unique label and its position
    for label in label_unique:
        label_pos[label] = []
    for x in range(label_array.shape[0]):
        for y in range(label_array.shape[1]):
            curr_label = label_array[x, y]
            if curr_label in label_unique:
                label_pos[curr_label].append([x, y])
    return label_pos


def divide_labels_by_ratio(label_pos, train_ratio):
    """
    # divide to training set and testing set for label
    :param label_pos:
    :param train_ratio:
    :return:
    """
    label_unique = label_pos.keys()
    training_num = dict()
    for label in label_unique:
        training_num[label] = int(len(label_pos[label]) * train_ratio)

    train_label_pos = dict()
    test_label_pos = dict()

    for curr_label in label_unique:
        curr_label_pos = np.random.permutation(label_pos[curr_label])
        train_label_pos[curr_label] = curr_label_pos[: int(
            training_num[curr_label])]
        test_label_pos[curr_label] = curr_label_pos[training_num[curr_label]:]
    return train_label_pos, test_label_pos


def divide_labels_by_nums(label_pos, train_nums_each_class):
    """
    :param label_pos: a dictionary for unique label and its position
    :param train_nums_each_class: each class has the same nums for training set
    :return:
    """
    label_unique = label_pos.keys()
    training_num = dict()
    for label in label_unique:
        training_num[label] = int(train_nums_each_class)

    train_label_pos = dict()
    test_label_pos = dict()

    for curr_label in label_unique:
        curr_label_pos = np.random.permutation(label_pos[curr_label])
        train_label_pos[curr_label] = curr_label_pos[: int(
            training_num[curr_label])]
        test_label_pos[curr_label] = curr_label_pos[training_num[curr_label]:]
    return train_label_pos, test_label_pos


def get_data_divided(train_label_pos, test_label_pos, data):
    """
    # divide to training set and testing set for data
    :param train_label_pos: training  label and its position
    :param test_label_pos:  test label and its position
    :param data: original data
    :return:
    """
    # get label,index,data of train
    train_nums = sum(len(train_label_pos[key])
                     for key in train_label_pos.keys())
    train_label, train_index = np.zeros(
        train_nums), np.zeros([train_nums, 2], dtype='int')
    train_data = np.zeros([train_nums, data.shape[2]])

    i = 0
    train_label_unique = train_label_pos.keys()
    for label in train_label_unique:
        curr_label_pos_array = train_label_pos[label]
        for curr_label_pos in range(len(curr_label_pos_array)):
            train_label[i] = label
            train_index[i] = curr_label_pos_array[curr_label_pos]
            train_data[i, :] = data[train_index[i][0], train_index[i][1], :]
            i = i + 1

    # get label,index,data of test
    test_nums = sum(len(test_label_pos[key]) for key in test_label_pos.keys())
    test_label, test_index = np.zeros(
        test_nums), np.zeros([test_nums, 2], dtype='int')
    test_data = np.zeros([test_nums, data.shape[2]])

    i = 0
    print(sorted(test_label_pos.keys()))
    test_label_unique = test_label_pos.keys()
    for label in test_label_unique:
        curr_label_pos_array = test_label_pos[label]
        for curr_label_pos in range(len(curr_label_pos_array)):
            test_label[i] = label
            test_index[i] = curr_label_pos_array[curr_label_pos]
            test_data[i, :] = data[test_index[i][0], test_index[i][1], :]
            i = i + 1

    return (train_label, train_index, train_data), (test_label, test_index, test_data)


def get_data_divided_by_3d(train_label_pos, test_label_pos, data):
    """
    # divide to training set and testing set for data
    :param train_label_pos: training  label and its position
    :param test_label_pos:  test label and its position
    :param data: original data
    :return:
    """
    # get label,index,data of train
    preprocess = pre.HSI_preprocess(name='RPCA', dst_shape=None)
    # transform the shape of data from (m, n, channels)
    # to (m, n, patch_size, patch_size, channels)
    patch_datas = preprocess.get_patch_data(data=data, patch_size=5)
    train_nums = sum(len(train_label_pos[key])
                     for key in train_label_pos.keys())
    train_label, train_index = np.zeros(
        train_nums), np.zeros([train_nums, 2], dtype='int')
    # n_samples x patch_size x patch_size x channels
    train_data = np.zeros([train_nums, ] + list(patch_datas.shape[-3:]))

    i = 0
    train_label_unique = train_label_pos.keys()
    for label in train_label_unique:
        curr_label_pos_array = train_label_pos[label]
        for curr_label_pos in range(len(curr_label_pos_array)):
            train_label[i] = label
            train_index[i] = curr_label_pos_array[curr_label_pos]
            train_data[i, :, :, :] = patch_datas[train_index[i]
                                                 [0], train_index[i][1], :, :, :]
            i = i + 1

    # get label,index,data of test
    test_nums = sum(len(test_label_pos[key]) for key in test_label_pos.keys())
    test_label, test_index = np.zeros(
        test_nums), np.zeros([test_nums, 2], dtype='int')
    # n_samples x patch_size x patch_size x channels
    test_data = np.zeros([test_nums, ] + list(patch_datas.shape[-3:]))

    i = 0
    print(sorted(test_label_pos.keys()))
    test_label_unique = test_label_pos.keys()
    for label in test_label_unique:
        curr_label_pos_array = test_label_pos[label]
        for curr_label_pos in range(len(curr_label_pos_array)):
            test_label[i] = label
            test_index[i] = curr_label_pos_array[curr_label_pos]
            test_data[i, :, :, :] = patch_datas[test_index[i]
                                                [0], test_index[i][1], :, :, :]
            i = i + 1
    train_data = np.array(train_data).swapaxes(1, 3).swapaxes(2, 3)
    test_data = np.array(test_data).swapaxes(1, 3).swapaxes(2, 3)
    return (train_label, train_index, train_data), (test_label, test_index, test_data)


def test_split_by_nums():
    import time
    import hyperspectral_datasets as HSI
    time1 = time.time()

    ip = HSI.HSIDataSet('indian_pines')
    ip.get_data()
    ip.get_labels()
    ip.pos = get_lables_pos(ip.labels, label_unique=[
                            2, 3, 5, 8, 10, 11, 12, 14])
    train_label_pos, test_label_pos = divide_labels_by_nums(ip.pos, 200)
    (train_label, train_index, traizn_data), (test_label, test_index, test_data) = get_data_divided(
        train_label_pos, test_label_pos, ip.data)
    print('train_label.shape : ', train_label.shape)
    print('train_index.shape : ', train_index.shape)
    print('train_data.shape : ', train_data.shape)
    print('test_label.shape : ', test_label.shape)
    print('test_index.shape : ', test_index.shape)
    print('test_data.shape : ', test_data.shape)
    time2 = time.time()
    print('use time : ', time2 - time1)


def test_split_by_ratio():
    import time
    import hyperspectral_datasets as HSI
    time1 = time.time()

    ip = HSI.HSIDataSet('indian_pines')
    ip.get_data()
    ip.get_labels()
    ip.pos = get_lables_pos(ip.labels, label_unique=[
                            2, 3, 5, 8, 10, 11, 12, 14])
    train_label_pos, test_label_pos = divide_labels_by_ratio(ip.pos, 0.5)
    (train_label, train_index, train_data), (test_label, test_index, test_data) = get_data_divided(
        train_label_pos, test_label_pos, ip.data)
    print('train_label.shape : ', train_label.shape)
    print('train_index.shape : ', train_index.shape)
    print('train_data.shape : ', train_data.shape)
    print('test_label.shape : ', test_label.shape)
    print('test_index.shape : ', test_index.shape)
    print('test_data.shape : ', test_data.shape)
    time2 = time.time()
    print('use time : ', time2 - time1)


def test_train_test_split():
    import time
    import hyperspectral_datasets as HSI
    time1 = time.time()

    ip = HSI.HSIDataSet('indian_pines')
    ip.get_data()
    ip.get_labels()
    label_unique = [2, 3, 5, 6, 8, 10, 11, 12, 14]
    (train_label, train_index, train_data), (test_label, test_index, test_data) = train_test_split(ip.data, ip.labels,
                                                                                                   label_unique=label_unique,
                                                                                                   train=200)
    print('train_label.shape : ', train_label.shape)
    print('train_index.shape : ', train_index.shape)
    print('train_data.shape : ', train_data.shape)
    print('test_label.shape : ', test_label.shape)
    print('test_index.shape : ', test_index.shape)
    print('test_data.shape : ', test_data.shape)
    time2 = time.time()
    print('use time : ', time2 - time1)


if __name__ == '__main__':
    test_split_by_nums()
    test_split_by_ratio()
    test_train_test_split()
