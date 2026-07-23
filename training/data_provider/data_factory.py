from data_provider.data_loader import get_data, LoadSeedvDataset, generate_dataset_tuab, get_seed4_data, get_wm_data_per_subject_split, get_wmsub_data
from data_provider.data_loader import get_bcic2020t3_data
import torch
from torch.utils.data import DataLoader


data_dict = {
    'bcic2a': get_data,
    'bcic2b': get_data,
    'seedv': LoadSeedvDataset,
    'tuab': generate_dataset_tuab,
    'tuev': generate_dataset_tuab,
    'seediv': get_seed4_data,
    'mmwm': get_wm_data_per_subject_split,
    'mmwm_small': get_wmsub_data,
    'bcic2020_T3': get_bcic2020t3_data
}

def data_provider(args, flag, sub):
    if args.data not in data_dict:
        raise ValueError(f"Unsupported dataset: {args.data}")

    data_fn = data_dict[args.data]

    if args.data == 'seedv':
        train_dataset, valid_dataset, test_dataset = data_fn(args.data_path)
    elif args.data == 'tuab':
        train_dataset, valid_dataset, test_dataset, ch_names, metrics = data_fn(dataset='TUAB', data_path=args.data_path,)
    elif args.data == 'tuev':
        train_dataset, valid_dataset, test_dataset, ch_names, metrics = data_fn(dataset='TUEV', data_path=args.data_path,)
    elif args.data == 'seediv':
        seediv_channels = [      'FP1', 'FPZ', 'FP2', 
                               'AF3', 'AF4', 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                      'PO7', 'PO3', 'POZ',  'PO4', 'PO8', 
                               'O1', 'OZ', 'O2', ]
        train_dataset, valid_dataset, test_dataset = data_fn(target_sub=sub, data_path=args.data_path, use_channels=seediv_channels)
    elif args.data == 'mmwm':
        mmwm_channels = [
            "FP1", "FP2",
            "F7", "F3", "FZ", "F4", "F8",
            "FC5", "FC1", "FC2", "FC6",
            "T7", "C3", "CZ", "C4", "T8",
            "CP5", "CP1", "CP2", "CP6",
            "P7", "P3", "PZ", "P4", "P8",
            "O1", "OZ", "O2"
        ]
        train_dataset,valid_dataset,test_dataset = get_wm_data_per_subject_split(data_path=args.data_path, split=(0.8, 0., 0.2), use_channels=mmwm_channels)
    elif args.data == 'mmwm_small':
        mmwm_channels = [
            "FP1", "FP2",
            "F7", "F3", "FZ", "F4", "F8",
            "FC5", "FC1", "FC2", "FC6",
            "T7", "C3", "CZ", "C4", "T8",
            "CP5", "CP1", "CP2", "CP6",
            "P7", "P3", "PZ", "P4", "P8",
            "O1", "OZ", "O2"
        ]
        train_dataset,valid_dataset,test_dataset = data_fn(data_path=args.data_path, sub=sub, use_channels=mmwm_channels, split=(0.8,0.,0.2))
    elif args.data == 'bcic2020_T3':
        train_dataset, valid_dataset, test_dataset = data_fn(args.data_path)
    else: 
        train_dataset, valid_dataset, test_dataset = data_fn(
            sub=sub,
            data_path=args.data_path,
            target_sample=1024,
            is_few_EA=True
        )

    if flag == 'train':
        data_set = train_dataset
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    elif flag == 'val':
        data_set = valid_dataset
        shuffle_flag = False
        drop_last = False
        batch_size = args.eval_batch_size
    elif flag == 'test':
        data_set = test_dataset
        shuffle_flag = False
        drop_last = False
        batch_size = args.eval_batch_size
    else:
        if args.data == 'seedv':
            train_loader =  torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,collate_fn=train_dataset.collate,shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.batch_size,collate_fn=valid_dataset.collate,shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batch_size,collate_fn=test_dataset.collate,shuffle=False)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
            test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=0, shuffle=False)
        return train_loader, valid_loader, test_loader

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=True
    )

    return data_set, data_loader