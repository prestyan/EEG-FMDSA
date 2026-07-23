import scipy
from scipy import signal
import os
import lmdb
import pickle
import numpy as np
import mne

SEEDV_CHANNEL_LIST = [
    'FP1', 'FPZ', 'FP2',
    'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

USE_CHANNELS_NAMES = [
    'FP1', 'FPZ', 'FP2',
    'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
    'O1', 'OZ', 'O2'
]

keep_indices = [SEEDV_CHANNEL_LIST.index(ch) for ch in USE_CHANNELS_NAMES]

useless_ch = ['M1', 'M2', 'VEO', 'HEO']
trials_of_sessions = {
    '1': {'start': [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
          'end': [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]},

    '2': {'start': [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
          'end': [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]},

    '3': {'start': [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
          'end': [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]},
}
labels_of_sessions = {
    '1': [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0, ],
    '2': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0, ],
    '3': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0, ],
}

root_dir = './EEG_raw'
files = [file for file in os.listdir(root_dir)]
files = sorted(files)
print(files)

trials_split = {
    'train': range(5),
    'val': range(5, 10),
    'test': range(10, 15),
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

window_size = 256 # 1024

db = lmdb.open('./processed_seedv2', map_size=15614542346)

for file in files:
    print(file)
    raw = mne.io.read_raw_cnt(os.path.join(root_dir, file), preload=True)
    raw.drop_channels(useless_ch)
    # raw.set_eeg_reference(ref_channels='average')
    raw.resample(256)
    raw.filter(l_freq=0.3, h_freq=75)
    raw_ch_names = [ch.upper() for ch in raw.ch_names]
    assert raw_ch_names == SEEDV_CHANNEL_LIST, \
        f"Channel order mismatch!\nraw: {raw_ch_names}\nexpected: {SEEDV_CHANNEL_LIST}"
    data_matrix = raw.get_data(units='uV') / 1000
    data_matrix = data_matrix[keep_indices]
    session_index = file.split('_')[1]
    data_trials = [
        data_matrix[:,
        trials_of_sessions[session_index]['start'][j] * 256:trials_of_sessions[session_index]['end'][j] * 256]
        for j in range(15)]
    labels = labels_of_sessions[session_index]
    for mode in trials_split.keys():
        for index in trials_split[mode]:
            data = data_trials[index]
            label = labels[index]
            valid_len = data.shape[1] // window_size * window_size
            data = data[:, :valid_len]
            data = data.reshape(58, -1, window_size)
            data = data.transpose(1, 0, 2)
            print(data.shape)
            for i, sample in enumerate(data):
                sample_key = f'{file}-{index}-{i}'
                data_dict = {
                    'sample': sample, 'label': label
                }
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[mode].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()