bcic2a_channels = [
    'FZ',
    'FC3', 'FC1', 'FCZ', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
    'P1', 'PZ', 'P2',
    'POZ'
]

bcic2b_channels = ["C3", "Cz", "C4"]

seedv_channels = [
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

original_tueg_channels = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3',
    'P4', 'O1', 'O2', 'F7', 'F8',
    'T3', 'T4', 'T5', 'T6',
    'A1', 'A2',
    'FZ', 'CZ', 'PZ',
    'T1', 'T2'
]

target_tueg_channels = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3',
    'P4', 'O1', 'O2', 'F7', 'F8',
    'T7', 'T8', 'P7', 'P8',
    'FZ', 'CZ', 'PZ'
]

seediv_channels = [      'FP1', 'FPZ', 'FP2', 
                               'AF3', 'AF4', 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                      'PO7', 'PO3', 'POZ',  'PO4', 'PO8', 
                               'O1', 'OZ', 'O2', ]

mmwm_channels = [
        "FP1", "FP2",
        "F7", "F3", "FZ", "F4", "F8",
        "FC5", "FC1", "FC2", "FC6",
        "T7", "C3", "CZ", "C4", "T8",
        "CP5", "CP1", "CP2", "CP6",
        "P7", "P3", "PZ", "P4", "P8",
        "O1", "OZ", "O2"
    ]

bcic2020t3_channels = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
        'P7', 'P3', 'PZ', 'P4', 'P8',
        'PO9', 'O1', 'OZ', 'O2', 'PO10',
        'AF7', 'AF3', 'AF4', 'AF8',
        'F5', 'F1', 'F2', 'F6',
        'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10',
        'C5', 'C1', 'C2', 'C6',
        'TP7', 'CP3', 'CPZ', 'CP4', 'TP8',
        'P5', 'P1', 'P2', 'P6',
        'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
    ]

target_bcic2020t3_channels = [
        'FP1', 'FPZ', 'FP2',
        'AF3', 'AF4',
        'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
        'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
        'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
        'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
        'O1', 'OZ', 'O2',
    ]

selected_bcic2020t3_channels = [
        ch for ch in target_bcic2020t3_channels
        if ch in bcic2020t3_channels
    ]

selected_bcic2020t3_idx = [
    bcic2020t3_channels.index(ch)
    for ch in selected_bcic2020t3_channels
]