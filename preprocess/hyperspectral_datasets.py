import os
import scipy.io as sio
import subprocess


def make_if_not_exist(dir=''):
    dir = os.path.expanduser(dir)
    if not os.path.exists(dir):
        subprocess.check_call("mkdir {}".format(dir), shell=True)


class DataSet:
    def __init__(self, name, data_dir=''):
        self.name = name
        self.dir = data_dir

        self.data = None
        self.data_file = None
        self.data_key = None

        self.labels = None
        self.label_file = None
        self.label_key = None

    def set_dir(self, data_dir):
        self.dir = data_dir

    def set_data_file(self, data_file_name):
        data_file = "{}/{}".format(self.dir, data_file_name)
        assert os.path.exists(data_file)

    def set_label_file(self, label_file_name):
        label_file = "{}/{}".format(self.dir, label_file_name)
        assert os.path.exists(label_file)

    def get_data(self):
        data_file = os.path.expanduser(
            "{}/{}".format(self.dir, self.data_file))
        assert os.path.exists(data_file)
        self.data = sio.loadmat(data_file)[self.data_key]

    def get_labels(self):
        label_file = os.path.expanduser(
            "{}/{}".format(self.dir, self.label_file))
        assert os.path.exists(label_file)
        self.labels = sio.loadmat(label_file)[self.label_key]


class HSIDataSetInfo():
    info = {}
    info['indian_pines'] = {
        'data_file_name': 'Indian_pines_corrected.mat',
        'data_key': 'indian_pines_corrected',
        'label_file_name': 'Indian_pines_gt.mat',
        'label_key': 'indian_pines_gt'
    }
    info['salina'] = {
        'data_file_name': 'Salinas_corrected.mat',
        'data_key': 'salinas_corrected',
        'label_file_name': 'Salinas_gt.mat',
        'label_key': 'salinas_gt'
    }
    info['pavia'] = {
        'data_file_name': 'Pavia.mat',
        'data_key': 'pavia',
        'label_file_name': 'Pavia_gt.mat',
        'label_key': 'pavia_gt'
    }
    info['paviau'] = {
        'data_file_name': 'PaviaU.mat',
        'data_key': 'paviaU',
        'label_file_name': 'PaviaU_gt.mat',
        'label_key': 'paviaU_gt'
    }


class HSIDataSet(DataSet, HSIDataSetInfo):
    def __init__(self, name):
        self.HSI_dir = os.path.expanduser('../hyperspectral_datas/')
        DataSet.__init__(self, name, data_dir=self.HSI_dir + name + '/data')
        self.set_info()

    def set_info(self):
        self.dataset_info = HSIDataSetInfo.info[self.name]
        self.data_file = self.dataset_info.get('data_file_name')
        self.data_key = self.dataset_info.get('data_key')
        self.label_file = self.dataset_info.get('label_file_name')
        self.label_key = self.dataset_info.get('label_key')


if __name__ == '__main__':
    dataset = HSIDataSet('indian_pines')
    dataset.get_data()
    dataset.get_labels()
