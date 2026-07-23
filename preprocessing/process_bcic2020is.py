# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import pandas as pd
import scipy.io as sio



data_root = "./"

train_dir = os.path.join(data_root, "Training set")
valid_dir = os.path.join(data_root, "Validation set")
test_dir = os.path.join(data_root, "Test set")

label_file = os.path.join(
    data_root,
    "Track3_Answer Sheet_Test.xlsx"
)

output_root = os.path.join(
    os.path.dirname(data_root),
    "preprocessed_BCIC2020_T3"
)

output_dirs = {
    "train": os.path.join(output_root, "train"),
    "valid": os.path.join(output_root, "valid"),
    "test": os.path.join(output_root, "test"),
}

for output_dir in output_dirs.values():
    os.makedirs(output_dir, exist_ok=True)



sfreq = 256

start_idx = int(0.5 * sfreq)       # 128

segment_length = int(2.0 * sfreq)  # 512

end_idx = start_idx + segment_length  # 640

train_files = sorted([
    file for file in os.listdir(train_dir)
    if file.lower().endswith(".mat")
])

valid_files = sorted([
    file for file in os.listdir(valid_dir)
    if file.lower().endswith(".mat")
])

test_files = sorted([
    file for file in os.listdir(test_dir)
    if file.lower().endswith(".mat")
])

print("Train files:", len(train_files))
print("Valid files:", len(valid_files))
print("Test files:", len(test_files))


for file in train_files:

    input_path = os.path.join(train_dir, file)
    output_path = os.path.join(output_dirs["train"], file)

    data = sio.loadmat(input_path)

    eeg = data["epo_train"][0][0][4]

    eeg = eeg.transpose(2, 1, 0)

    labels = data["epo_train"][0][0][5].transpose(1, 0)

    # one-hot -> 0~4
    labels = np.argmax(labels, axis=1)

    eeg = eeg[:, :, start_idx:end_idx]

    eeg = eeg.astype(np.float32)
    labels = labels.astype(np.int64).reshape(-1, 1)

    assert eeg.shape == (300, 64, 512), (
        f"{file} EEG shape error: {eeg.shape}"
    )
    assert labels.shape == (300, 1), (
        f"{file} label shape error: {labels.shape}"
    )

    sio.savemat(
        output_path,
        {
            "x_data": eeg,
            "y_data": labels,
        },
        do_compression=True,
    )

    print(
        f"[train] {file}: "
        f"x_data={eeg.shape}, y_data={labels.shape}"
    )


for file in valid_files:

    input_path = os.path.join(valid_dir, file)
    output_path = os.path.join(output_dirs["valid"], file)

    data = sio.loadmat(input_path)

    eeg = data["epo_validation"][0][0][4]

    eeg = eeg.transpose(2, 1, 0)

    labels = data["epo_validation"][0][0][5].transpose(1, 0)

    # one-hot -> 0~4
    labels = np.argmax(labels, axis=1)

    eeg = eeg[:, :, start_idx:end_idx]

    eeg = eeg.astype(np.float32)
    labels = labels.astype(np.int64).reshape(-1, 1)

    assert eeg.shape == (50, 64, 512), (
        f"{file} EEG shape error: {eeg.shape}"
    )
    assert labels.shape == (50, 1), (
        f"{file} label shape error: {labels.shape}"
    )

    sio.savemat(
        output_path,
        {
            "x_data": eeg,
            "y_data": labels,
        },
        do_compression=True,
    )

    print(
        f"[valid] {file}: "
        f"x_data={eeg.shape}, y_data={labels.shape}"
    )



df = pd.read_excel(
    label_file,
    header=None,
)

subject_labels = []

for col_idx in range(2, 31, 2):

    labels = pd.to_numeric(
        df.iloc[:, col_idx],
        errors="coerce",
    )

    labels = labels[
        labels.isin([1, 2, 3, 4, 5])
    ].to_numpy(dtype=np.int64)

    if len(labels) != 50:
        raise ValueError(
            f"Excel column {col_idx} should contain 50 labels, "
            f"but found {len(labels)}."
        )

    subject_labels.append(labels)

# [15, 50]
all_labels = np.stack(subject_labels, axis=0)

all_labels = all_labels - 1
print(all_labels)

print("Test labels shape:", all_labels.shape)

for sub_idx in range(15):
    print(
        f"Subject {sub_idx + 1:02d}:",
        np.unique(all_labels[sub_idx], return_counts=True),
    )

assert all_labels.shape == (15, 50)
assert np.isin(all_labels, [0, 1, 2, 3, 4]).all()


for subject_idx, file in enumerate(test_files):

    input_path = os.path.join(test_dir, file)
    output_path = os.path.join(output_dirs["test"], file)

    with h5py.File(input_path, "r") as data:

        # [trials, channels, time]
        eeg = np.asarray(data["epo_test"]["x"])

    labels = all_labels[subject_idx]

    if eeg.shape[0] != 50 or eeg.shape[1] != 64:

        # 若读取结果为 [time, channels, trials]
        if eeg.shape[1] == 64 and eeg.shape[2] == 50:
            eeg = eeg.transpose(2, 1, 0)
        else:
            raise ValueError(
                f"{file} unexpected EEG shape: {eeg.shape}"
            )

    eeg = eeg[:, :, start_idx:end_idx]

    eeg = eeg.astype(np.float32)
    labels = labels.astype(np.int64).reshape(-1, 1)

    assert eeg.shape == (50, 64, 512), (
        f"{file} EEG shape error: {eeg.shape}"
    )
    assert labels.shape == (50, 1), (
        f"{file} label shape error: {labels.shape}"
    )

    sio.savemat(
        output_path,
        {
            "x_data": eeg,
            "y_data": labels,
        },
        do_compression=True,
    )

    print(
        f"[test] {file}: "
        f"x_data={eeg.shape}, y_data={labels.shape}"
    )


print("\nPreprocessing completed.")
print("Output folder:", output_root)