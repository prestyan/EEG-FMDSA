import zlib

import torch
import os
import sys
import numpy as np
import random
import scipy.io as sio
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
from einops import rearrange
import copy
import gc

import scipy
from data_provider.sea import StableEA, compute_R_ref, EA

from torch.utils.data import Dataset,DataLoader

from scipy import stats
from scipy.signal import resample
from torch.utils.data import ConcatDataset

import glob
import torch
# from Modules.spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified,SPDVectorize
# from Data_process.process_function import Load_BCIC_2a_raw_data
from collections import Counter
current_path = os.path.abspath('./')
root_path = current_path # os.path.split(current_path)[0]

sys.path.append(root_path)



class eeg_dataset(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self,feature,label,subject_id=None):
        super(eeg_dataset,self).__init__()

        self.x = feature
        self.y = label
        self.s = subject_id

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if self.s is not None:
            return self.x[index], self.y[index], self.s[index]
        return self.x[index], self.y[index]
    
    def get_num_class(self, num_class=[1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            label = self.y[i]
            label = int(label)
            if num_class[label]>0:
                num_class[label]-=1
                res[label].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y     
    
    def get_num_subject(self, num_class=[1,1,1,1,1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            s = self.s[i]
            s = int(s)
            if num_class[s]>0:
                num_class[s]-=1
                res[s].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y        

def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    # print(x.shape)
    # squeeze and unsqueeze because these are done before batching
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    # Supports batch dimension
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

def get_data(sub,data_path,few_shot_number = 1, is_few_EA = False, target_sample=-1, use_avg=True, use_channels=None):
    # 1) 先算全局参考（只用 source train）
    # R_ref = None
    # if is_few_EA:
    #     R_ref = compute_R_ref(data_path, target_sub=sub)
    
    target_session_1_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(sub))
    target_session_2_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(sub))

    session_1_data = sio.loadmat(target_session_1_path)
    session_2_data = sio.loadmat(target_session_2_path)
    R = None
    if is_few_EA is True:
        session_1_x = EA(session_1_data['x_data'],R)
    else:
        session_1_x = session_1_data['x_data']
        
    if is_few_EA is True:
        session_2_x = EA(session_2_data['x_data'],R)
    else:
        session_2_x = session_2_data['x_data']
    
    # -- debug for BCIC 2b
    test_x_1 = torch.FloatTensor(session_1_x)      
    test_y_1 = torch.LongTensor(session_1_data['y_data']).reshape(-1)

    test_x_2 = torch.FloatTensor(session_2_x)      
    test_y_2 = torch.LongTensor(session_2_data['y_data']).reshape(-1)
    session_1_x_sub = session_2_x
    session_1_y_sub = test_y_2

    test_s = torch.full((len(test_y_1) + len(test_y_2),),8,dtype=torch.long)
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample, use_avg=use_avg)
        test_x_2 = temporal_interpolation(test_x_2, target_sample, use_avg=use_avg)
    if use_channels is not None:
        test_dataset = eeg_dataset(torch.cat([test_x_1,test_x_2],dim=0)[:,use_channels,:],torch.cat([test_y_1,test_y_2], dim=0), test_s)
    else:
        test_dataset = eeg_dataset(torch.cat([test_x_1,test_x_2],dim=0),torch.cat([test_y_1,test_y_2], dim=0), test_s)

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0
    for i in range(1,10):
        if i == sub:
            # source_train_x.extend(session_1_x_sub)
            # source_train_y.extend(session_1_y_sub)
            # source_train_s.append(test_s)
            continue
        train_path = os.path.join(data_path,r'sub{}_train/Data.mat'.format(i))
        train_data = sio.loadmat(train_path)
    
        test_path = os.path.join(data_path,r'sub{}_test/Data.mat'.format(i))
        test_data = sio.loadmat(test_path)
        if is_few_EA is True:
            session_1_x = EA(train_data['x_data'],R)
        else:
            session_1_x = train_data['x_data']

        session_1_y = train_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_1_x,session_1_y,test_size = 0.1,stratify = session_1_y)
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)

        if is_few_EA is True:
            session_2_x = EA(test_data['x_data'],R)
        else:
            session_2_x = test_data['x_data']

        session_2_y = test_data['y_data'].reshape(-1)

        train_x,valid_x,train_y,valid_y = train_test_split(session_2_x,session_2_y,test_size = 0.1,stratify = session_2_y)
        
        source_train_x.extend(train_x)
        source_train_y.extend(train_y)
        source_train_s.append(torch.ones((len(train_y),))*subject_id)

        source_valid_x.extend(valid_x)
        source_valid_y.extend(valid_y)
        source_valid_s.append(torch.ones((len(valid_y),))*subject_id)
        subject_id+=1
    
    source_train_x = torch.FloatTensor(np.array(source_train_x))
    source_train_y = torch.LongTensor(np.array(source_train_y))
    source_train_s = torch.cat(source_train_s, dim=0)

    source_valid_x = torch.FloatTensor(np.array(source_valid_x))
    source_valid_y = torch.LongTensor(np.array(source_valid_y))
    source_valid_s = torch.cat(source_valid_s, dim=0)
    
    if target_sample>0:
        source_train_x = temporal_interpolation(source_train_x, target_sample, use_avg=use_avg)
        source_valid_x = temporal_interpolation(source_valid_x, target_sample, use_avg=use_avg)
        
    if use_channels is not None:
        train_dataset = eeg_dataset(source_train_x[:,use_channels,:],source_train_y,source_train_s)
    else:
        train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)
    
    if use_channels is not None:
        valid_datset = eeg_dataset(source_valid_x[:,use_channels,:],source_valid_y,source_valid_s)
    else:
        valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)
    
    return train_dataset,valid_datset,test_dataset

def to_tensor(array):
    return torch.from_numpy(array).float()


import pickle
import lmdb
_LMDB_ENV_CACHE = {}
_LMDB_CACHE_PID = os.getpid()


def get_lmdb_env(data_dir):
    """
    Ensure that each process opens each LMDB path only once.

    Compatible with:
    - train/val/test sharing the same LMDB
    - repeated calls to data_provider()
    - DataLoader num_workers=0
    - DataLoader num_workers>0
    """
    global _LMDB_ENV_CACHE
    global _LMDB_CACHE_PID

    current_pid = os.getpid()
    if current_pid != _LMDB_CACHE_PID:
        _LMDB_ENV_CACHE = {}
        _LMDB_CACHE_PID = current_pid

    data_dir = os.path.realpath(os.path.abspath(data_dir))

    if data_dir not in _LMDB_ENV_CACHE:
        _LMDB_ENV_CACHE[data_dir] = lmdb.open(
            data_dir,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_spare_txns=0,
        )

    return _LMDB_ENV_CACHE[data_dir]

class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode='train',
    ):
        super(CustomDataset, self).__init__()
        # self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        self.db = get_lmdb_env(data_dir)
        with self.db.begin(write=False) as txn:
            all_keys = pickle.loads(txn.get('__keys__'.encode()))
            if mode == 'train':
                self.keys = list(all_keys[mode]) + list(all_keys['val'])
            else:
                self.keys = list(all_keys[mode])

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        label = pair['label']
        return data, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label).long()


def LoadSeedvDataset(datasets_dir):
    train_set = CustomDataset(datasets_dir, mode='train')
    val_set = CustomDataset(datasets_dir, mode='val')
    test_set = CustomDataset(datasets_dir, mode='test')
    print(len(train_set), len(val_set), len(test_set))
    print(len(train_set) + len(val_set) + len(test_set))
    return train_set, val_set, test_set

class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y

class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y
    
from collections import Counter

def prepare_TUAB_dataset(root):
    # set random seed
    seed = 2025
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_dataset = TUABLoader(os.path.join(root, "train"), train_files)
    test_dataset = TUABLoader(os.path.join(root, "test"), test_files)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files)
    # train_dataset = ConcatDataset([train_dataset, test_dataset])
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset

def prepare_TUEV_dataset(root):
    # set random seed
    seed = 2025
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_test"))

    # prepare training and test data loader
    train_dataset = TUEVLoader(
        os.path.join(
            root, "processed_train"), train_files
    )
    test_dataset = TUEVLoader(
        os.path.join(
            root, "processed_test"), test_files
    )
    val_dataset = TUEVLoader(
        os.path.join(
            root, "processed_eval"), val_files
    )
    print(len(train_files), len(val_files), len(test_files))
    all_labels = []
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        label = sample[-1]
        all_labels.append(int(label))

    label_counts = Counter(all_labels)
    print("Class counts:", label_counts)
    num_classes = max(label_counts.keys()) + 1
    total_samples = len(all_labels)

    class_weights = []

    for cls in range(num_classes):
        count = label_counts.get(cls, 0)

        if count == 0:
            raise ValueError(f"Class {cls} has no samples in train_dataset.")

        weight = total_samples / (num_classes * count)
        class_weights.append(weight)

    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    print("Class weights:", class_weights)
    return train_dataset, test_dataset, val_dataset

def generate_dataset_tuab(dataset='TUAB',
                           data_path='F:/data/TUAB/tuh_eeg_abnormal/edf/processed',
                           index=1, batch_size=64):
    assert dataset in ['TUAB', 'TUEV']
    train_dataset, test_dataset, val_dataset = None, None, None
    ch_names, metrics = None, None
    if dataset == 'TUAB':
        train_dataset, test_dataset, val_dataset = prepare_TUAB_dataset(data_path)
        ch_names = ['EEG FP1', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                    'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        # nb_classes = 1
        metrics = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]

    elif dataset == 'TUEV':
        train_dataset, test_dataset, val_dataset = prepare_TUEV_dataset(data_path)
        ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                    'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF',
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

        ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        # nb_classes = 6
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    '''
    elif dataset == 'TUSZ':
        train_dataset, test_dataset, val_dataset = prepare_TUSZ_dataset(None)

        # ch_names = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
        #             'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
        # ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
        ch_names = ['FP1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'FZ', 'CZ', 'PZ', 'FP2', 'F4', 'C4', 'P4', 'F8',
                    'T4', 'T6', 'O2']
        # nb_classes = 8
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
    '''
    return train_dataset, test_dataset, val_dataset, ch_names, metrics

class eeg_dataset(Dataset):
    '''
    A class need to input the Dataloader in the pytorch.
    '''
    def __init__(self,feature,label,subject_id=None):
        super(eeg_dataset,self).__init__()

        self.x = feature
        self.y = label
        self.s = subject_id

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def get_num_class(self, num_class=[1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            label = self.y[i]
            label = int(label)
            if num_class[label]>0:
                num_class[label]-=1
                res[label].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y     
    
    def get_num_subject(self, num_class=[1,1,1,1,1,1,1,1]):
        res = [[] for i in num_class]
        idxs = [i for i in range(len(self.y))]
        while sum(num_class)>0:
            i = random.choice(idxs)
            s = self.s[i]
            s = int(s)
            if num_class[s]>0:
                num_class[s]-=1
                res[s].append((self.x[i],self.y[i]))
            
        re2= []
        for r in res:
            re2.extend(r)
        x = torch.stack([x[0] for x in re2], dim=0)
        y = torch.stack([x[1] for x in re2], dim=0)
        
        return x, y        
    
def get_seed4_data(target_sub, data_path, valid_ratio=0., is_few_EA=False, use_channels=None):
    """
    LOSO-style split:
    - target_sub: test subject id (int)
    - data_path: directory containing 1_i.mat
    """
    original_channels = [
        # ===== Frontal pole / Anterior frontal =====
        'FP1', 'FPZ', 'FP2',
        'AF3',        'AF4',

        # ===== Frontal =====
        'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',

        # ===== Fronto-temporal / Fronto-central =====
        'FT7',
        'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
        'FT8',

        # ===== Temporal / Central =====
        'T7',
        'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
        'T8',

        # ===== Temporo-parietal / Centro-parietal =====
        'TP7',
        'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
        'TP8',

        # ===== Parietal =====
        'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',

        # ===== Parieto-occipital =====
        'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',

        # ===== Occipital / Cerebellar =====
        'CB1',
        'O1', 'OZ', 'O2',
        'CB2',
    ]

    train_data_list = []
    train_label_list = []

    test_data = None
    test_label = None

    R = None

    # ---- 1. 扫描所有被试文件 ----
    mat_files = sorted(glob.glob(os.path.join(data_path, "*.mat")))

    if len(mat_files) == 0:
        raise RuntimeError(f"No .mat files found in {data_path}")

    for mat_path in mat_files:
        name = os.path.basename(mat_path)
        # 假设文件名是 1_i.mat
        sub_id = int(os.path.splitext(name)[0].split("_")[1])

        mat = sio.loadmat(mat_path)
        data = mat["data"]    # (N, 62, 1024)
        if is_few_EA:
            data = EA(data, R)
        label = mat["label"].squeeze()  # (N,)

        if sub_id == target_sub:
            test_data = data
            test_label = label
            continue
        
        train_data_list.append(data)
        train_label_list.append(label)

    if test_data is None:
        raise ValueError(f"Target subject {target_sub} not found")

    # ---- 2. 合并训练被试 ----
    train_data_all = np.concatenate(train_data_list, axis=0)
    train_label_all = np.concatenate(train_label_list, axis=0)

    # 在这里补充，如果use_channels不为None，则从original_channels选择在use_channels中的通道
    # 并修改train_data_all, test_data
    if use_channels is not None:
        channel_indices = [original_channels.index(ch) for ch in use_channels if ch in original_channels]
        train_data_all = train_data_all[:, channel_indices, :]
        test_data = test_data[:, channel_indices, :]

    # ---- 3. 从训练集中划分验证集 ----
    if valid_ratio > 0.0:
        X_train, X_valid, y_train, y_valid = train_test_split(
            train_data_all,
            train_label_all,
            test_size=valid_ratio,
            stratify=train_label_all
        )
    else:
        X_train = train_data_all
        X_valid = np.array([])
        y_train = train_label_all
        y_valid = np.array([])

    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train).view(-1)

    test_data  = torch.FloatTensor(test_data)
    test_label  = torch.LongTensor(test_label).view(-1)

    if X_valid is not None:
        X_valid = torch.FloatTensor(X_valid)
        y_valid = torch.LongTensor(y_valid).view(-1)

    # ---- 4. 构建 Dataset ----
    train_dataset = eeg_dataset(X_train, y_train)
    print(X_train.shape)
    if len(X_valid) > 0:
        valid_dataset = eeg_dataset(X_valid, y_valid)
    else :
        valid_dataset = None
    test_dataset  = eeg_dataset(test_data, test_label)
    print(test_data.shape)

    return train_dataset, valid_dataset, test_dataset



def get_default_10_20_32ch_names():
    ch32 = [
        "FP1", "FP2",
        "F7", "F3", "FZ", "F4", "F8",
        "FC5", "FC1", "FC2", "FC6",
        "T7", "C3", "CZ", "C4", "T8",
        "CP5", "CP1", "CP2", "CP6",
        "P7", "P3", "PZ", "P4", "P8",
        "O1", "OZ", "O2",
        "A1", "A2",   # 常见的参考（或耳垂/乳突）
        "AFz"         # 有些帽子会有 AFz/POz 等（这里占位为 32）
    ]
    return ch32

def get_default_30ch_names():
    ch32 = get_default_10_20_32ch_names()
    DROP_CHS = {"A1", "A2", "AFz"}
    ch30 = [c for c in ch32 if c not in DROP_CHS]

    return ch30

def _normalize_name(name: str) -> str:
    name = name.strip()
    alias = {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
        "Pz": "PZ",
        "Fz": "FZ",
        "Cz": "CZ",
        "Oz": "OZ",
    }
    std = name[0].upper() + name[1:]
    std = std.replace("z", "Z")
    return alias.get(std, std)

def print_label_stats(y, name):
    labels, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    print(f"\n{name} label distribution:")
    for l, c in zip(labels, counts):
        print(f"  label {l}: {c} ({c/total:.2%})")
    print(f"  total: {total}")

def save_split_index(save_path, idx_tr, idx_va, idx_te, one_based=False):
    if one_based:
        idx_tr = idx_tr + 1
        idx_va = idx_va + 1
        idx_te = idx_te + 1

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("train:\n")
        f.write(" ".join(map(str, idx_tr.tolist())) + "\n\n")

        f.write("valid:\n")
        f.write(" ".join(map(str, idx_va.tolist())) + "\n\n")

        f.write("test:\n")
        f.write(" ".join(map(str, idx_te.tolist())) + "\n")


def get_wm_data_per_subject_split(data_path, use_channels=None, seed=2025, split=(0.6, 0.2, 0.2)):
    assert abs(sum(split) - 1.0) < 1e-6

    mat_files = sorted(glob.glob(os.path.join(data_path, "*.mat")))
    if len(mat_files) == 0:
        raise FileNotFoundError(f"在 {data_path} 下没有找到任何 .mat 文件")

    ch_names_30 = [_normalize_name(c) for c in get_default_30ch_names()]

    if use_channels is not None:
        use_channels_norm = [_normalize_name(c) for c in use_channels]
        name_to_idx = {c: i for i, c in enumerate(ch_names_30)}
        missing = [c for c in use_channels_norm if c not in name_to_idx]
        if missing:
            raise ValueError(f"use_channels 中有通道不在默认 30 通道列表里：{missing}\n默认通道列表为：{ch_names_30}")
        pick_idx = [name_to_idx[c] for c in use_channels_norm]
        picked_names = [ch_names_30[i] for i in pick_idx]
    else:
        pick_idx = None
        picked_names = ch_names_30

    X_tr_list, y_tr_list = [], []
    X_va_list, y_va_list = [], []
    X_te_list, y_te_list = [], []

    label_map = {11: 0, 14: 1}

    for fp in mat_files:
        mat = sio.loadmat(fp)
        if "data1" not in mat or "label1" not in mat:
            raise KeyError(f"{os.path.basename(fp)} 缺少 data1 或 label1")

        data = mat["data1"]          # [30, 500, 240]
        label = mat["label1"]        # [240, 1] 或 [240]

        if data.ndim != 3:
            raise ValueError(f"{os.path.basename(fp)} 的 data1 维度应为 3，实际为 {data.shape}")

        C, T, N = data.shape
        if C != 30:
            raise ValueError(f"{os.path.basename(fp)} 的 data1 第一维应为 30 通道，实际为 {C}")

        label = np.asarray(label).reshape(-1).astype(int)

        label = np.vectorize(label_map.get)(label).astype(np.int64)

        if label.shape[0] != N:
            raise ValueError(f"{os.path.basename(fp)} label1 样本数 {label.shape[0]} 与 data1 样本数 {N} 不一致")

        if pick_idx is not None:
            data = data[pick_idx, :, :]   # [C_sel, T, N]

        data = np.transpose(data, (2, 0, 1)).astype(np.float32)

        sub_seed = (abs(hash(os.path.basename(fp))) + seed) % (2**32)
        rng = np.random.default_rng(sub_seed)

        idx = np.arange(N)
        rng.shuffle(idx)

        n_train = int(round(split[0] * N))
        n_valid = int(round(split[1] * N))
        n_test = N - n_train - n_valid

        idx_tr = idx[:n_train]
        idx_va = idx[n_train:n_train + n_valid]
        idx_te = idx[n_train + n_valid:]

        X_tr_list.append(data[idx_tr]); y_tr_list.append(label[idx_tr])
        X_va_list.append(data[idx_va]); y_va_list.append(label[idx_va])
        X_te_list.append(data[idx_te]); y_te_list.append(label[idx_te])

        # filename = os.path.basename(fp)
        # sub_name = os.path.splitext(filename)[0]
        # os.makedirs('./mmwm_split', exist_ok=True)
        # split_save_path = os.path.join('./mmwm_split', f"{sub_name}.txt")
        # save_split_index(split_save_path, idx_tr, idx_va, idx_te, one_based=False)

    X_train = np.concatenate(X_tr_list, axis=0)
    y_train = np.concatenate(y_tr_list, axis=0)
    X_valid = np.concatenate(X_va_list, axis=0)
    y_valid = np.concatenate(y_va_list, axis=0)
    X_test  = np.concatenate(X_te_list, axis=0)
    y_test  = np.concatenate(y_te_list, axis=0)

    print_label_stats(y_train, "Train")
    print_label_stats(y_valid, "Valid")
    print_label_stats(y_test,  "Test")

    info = {
        "channels": picked_names,
        "split": split,
        "seed": seed,
        "files": [os.path.basename(p) for p in mat_files],
        "train_samples": int(len(y_train)),
        "valid_samples": int(len(y_valid)),
        "test_samples": int(len(y_test)),
    }
    print(info)

    train_dataset = eeg_dataset(X_train, y_train)
    valid_dataset = eeg_dataset(X_valid, y_valid)
    test_dataset  = eeg_dataset(X_test,  y_test)

    return train_dataset, valid_dataset, test_dataset

def get_wmsub_data(data_path, sub, use_channels=None, seed=2025, split=(0.6, 0.2, 0.2)):
    ch_names_30 = [_normalize_name(c) for c in get_default_30ch_names()]
    if use_channels is not None:
        use_channels_norm = [_normalize_name(c) for c in use_channels]
        name_to_idx = {c: i for i, c in enumerate(ch_names_30)}

        missing = [c for c in use_channels_norm if c not in name_to_idx]
        if missing:
            raise ValueError(
                f"use_channels 中有通道不在默认 30 通道列表里：{missing}\n"
                f"默认通道列表为：{ch_names_30}"
            )
        pick_idx = [name_to_idx[c] for c in use_channels_norm]
        picked_names = [ch_names_30[i] for i in pick_idx]
    else:
        pick_idx = None
        picked_names = ch_names_30
    fp = os.path.join(data_path, f"Sub_S1_{sub}.mat")
    mat = sio.loadmat(fp)
    data = mat["data1"]
    label = mat["label1"]
    label = np.asarray(label).reshape(-1)
    label_map = {11: 0, 14: 1}
    label = np.vectorize(label_map.get)(label)
    if pick_idx is not None:
        data = data[pick_idx, :, :]  # [C_sel, T, N]
    data = np.transpose(data, (2, 0, 1))  # [N, C, T]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(label))
    rng.shuffle(idx)
    n_total = len(idx)
    n_train = int(round(split[0] * n_total))
    n_valid = int(round(split[1] * n_total))

    n_test = n_total - n_train - n_valid

    idx_train = idx[:n_train]
    idx_valid = idx[n_train:n_train + n_valid]
    idx_test = idx[n_train + n_valid:]

    X_train, y_train = data[idx_train], label[idx_train]
    X_valid, y_valid = data[idx_valid], label[idx_valid]
    X_test, y_test = data[idx_test], label[idx_test]

    print_label_stats(y_train, "Train")
    print_label_stats(y_valid, "Valid")
    print_label_stats(y_test,  "Test")

    train_dataset = eeg_dataset(X_train, y_train)
    valid_dataset = eeg_dataset(X_valid, y_valid)
    test_dataset = eeg_dataset(X_test, y_test)

    return train_dataset, valid_dataset, test_dataset



def get_bcic2020t3_data(data_path):
    datasets = []

    for split in ["train", "valid", "test"]:
        split_path = os.path.join(data_path, split)
        files = sorted([
            file for file in os.listdir(split_path)
            if file.endswith(".mat")
        ])

        all_x = []
        all_y = []
        all_subject_id = []

        for subject_id, file in enumerate(files):
            data = sio.loadmat(
                os.path.join(split_path, file)
            )

            x = data["x_data"].astype(np.float32)
            y = data["y_data"].reshape(-1).astype(np.int64)

            all_x.append(x)
            all_y.append(y)

            all_subject_id.append(
                np.full(len(y), subject_id, dtype=np.int64)
            )

        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        all_subject_id = np.concatenate(all_subject_id, axis=0)

        dataset = eeg_dataset(
            feature=torch.from_numpy(all_x).float(),
            label=torch.from_numpy(all_y).long(),
            subject_id=torch.from_numpy(all_subject_id).long(),
        )

        print(
            f"{split}: "
            f"x={all_x.shape}, "
            f"y={all_y.shape}, "
            f"subject={all_subject_id.shape}"
        )

        datasets.append(dataset)

    train_dataset, valid_dataset, test_dataset = datasets

    return train_dataset, valid_dataset, test_dataset