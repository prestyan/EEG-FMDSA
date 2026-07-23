import math
import torch
import random
import os
import numpy as np
from sea import EA
import scipy.io as sio
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def InfoNCELoss(pred, target, t=0.):
    # B, NC, D
    NC, B, D = pred.shape
    similarity = torch.matmul(pred, target.transpose(1,2)) * math.exp(t)#  NC, B, B
    
    label = torch.arange(B).repeat(repeats=(NC,)).to(pred.device)
    
    logit1 = similarity.view((NC*B, B))
    accuracy1 = ((torch.argmax(logit1, dim=-1)==label)*1.0).mean()
    loss1 = torch.nn.functional.cross_entropy(logit1, label)
    
    logit2 = similarity.transpose(1,2).contiguous().view((NC*B, B))
    accuracy2 = ((torch.argmax(logit2, dim=-1)==label)*1.0).mean()
    loss2 = torch.nn.functional.cross_entropy(logit2, label)
    
    return (loss1+loss2)/2, (accuracy1 + accuracy2)/2

def BatchMAE_InfoNCELoss(pred, target, t=0.):
    # B, N, D
    pred1    = pred - pred.mean(dim=0)
    target1  = target - target.mean(dim=0)
    loss1, accuracy1 = InfoNCELoss(pred1, target1)
    
    pred2    = pred.transpose(0,1) # N,B,D
    target2  = target.transpose(0,1) 
    
    loss2, accuracy2 = InfoNCELoss(pred2, target2)
    
    return loss1, loss2, accuracy1, accuracy2

def CoupleInfoNCELoss(pred, target, t=0.):
    
    pred2    = pred.flatten(1,2).transpose(0,1) # NC,B,D
    target2  = target.flatten(1,2).transpose(0,1) 
    
    loss2, accuracy2 = InfoNCELoss(pred2, target2)
    
    pred1    = pred.transpose(1,2).flatten(0,1)# BC, N, D
    target1  = target.transpose(1,2).flatten(0,1)
    loss1, accuracy1 = InfoNCELoss(pred1, target1)
    
    pred3    = pred - pred.mean(dim=0)
    pred3    = pred3.flatten(0,1) # BN, C, D
    # pred3    = pred.flatten(0,1).transpose(0,2) # D, C, BN
    # pred3    = torch.layer_norm(pred3, normalized_shape=pred3.shape[-1:]).transpose(0,2) # BN, C, D
    
    
    # target3  = target.flatten(0,1).transpose(0,2) # D, C, BN
    # target3  = torch.layer_norm(target3, normalized_shape=target3.shape[-1:]).transpose(0,2)
    
    target3  = target - target.mean(dim=0)
    target3  = target3.flatten(0,1)
    
    loss3, accuracy3 = InfoNCELoss(pred3, target3)
    return loss1, loss2, loss3, accuracy1, accuracy2, accuracy3
    
def _generate_negatives(z, num_negatives=20):
    """Generate negative samples to compare each sequence location against"""
    batch_size, feat, full_len = z.shape
    z_k = z.permute([0, 2, 1]).reshape(-1, feat)
    with torch.no_grad():
        # candidates = torch.arange(full_len).unsqueeze(-1).expand(-1, self.num_negatives).flatten()
        negative_inds = torch.randint(0, full_len, size=(batch_size, full_len * num_negatives))
        # From wav2vec 2.0 implementation, I don't understand
        # negative_inds[negative_inds >= candidates] += 1

        for i in range(1, batch_size):
            negative_inds[i] += i * full_len

    z_k = z_k[negative_inds.view(-1)].view(batch_size, full_len, num_negatives, feat)
    return z_k, negative_inds

def _calculate_similarity(z, c, negatives, temp=0.1):
    c = c.permute([0, 2, 1]).unsqueeze(-2) # b, s, 1, t
    z = z.permute([0, 2, 1]).unsqueeze(-2)

    # negatives
    negative_in_target = (c == negatives).all(dim=-1) | (z == negatives).all(dim=-1)

    targets = torch.cat([c, negatives], dim=-2)
    # print(targets.shape)
    logits = torch.nn.functional.cosine_similarity(z, targets, dim=-1) / temp

    if negative_in_target.any():
        # print(negative_in_target.shape, logits.shape)
        logits[:,:,1:][negative_in_target] = float("-inf")
    
    return logits.view(-1, logits.shape[-1])

class SelfSuperviseLoss():
    def __init__(self, device='cuda', beta=1.0, num_negatives=10) -> None:
        self.beta = beta
        self.device = device
        self.num_negatives = num_negatives
        self.loss_fn = torch.nn.CrossEntropyLoss().to(device)

    def __call__(self, dec_data, enc_data):
        negatives, _ = _generate_negatives(enc_data, num_negatives=self.num_negatives)
        # Prediction -> batch_size x predict_length x predict_length
        logits = _calculate_similarity(enc_data, dec_data, negatives)
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        return self.loss_fn(logits, labels) + self.beta * enc_data.pow(2).mean(), logits 
    
    

class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats

class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr


class CosineWDSchedule(object):

    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.
    ):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd
    
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
    
    if target_sample>0:
        test_x_1 = temporal_interpolation(test_x_1, target_sample, use_avg=use_avg)
        test_x_2 = temporal_interpolation(test_x_2, target_sample, use_avg=use_avg)
    if use_channels is not None:
        test_dataset = eeg_dataset(torch.cat([test_x_1,test_x_2],dim=0)[:,use_channels,:],torch.cat([test_y_1,test_y_2],dim=0))
    else:
        test_dataset = eeg_dataset(torch.cat([test_x_1,test_x_2],dim=0),torch.cat([test_y_1,test_y_2],dim=0))

    source_train_x = []
    source_train_y = []
    source_train_s = []
    
    source_valid_x = []
    source_valid_y = []
    source_valid_s = []
    subject_id = 0
    for i in range(1,10):
        if i == sub:
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
        
    # source_train_x = torch.cat([source_train_x, test_x_1], dim=0)
    # source_train_y = torch.cat([source_train_y, test_y_1], dim=0)
    # source_train_x = torch.cat([source_train_x, test_x_2], dim=0)
    # source_train_y = torch.cat([source_train_y, test_y_2], dim=0)

    if use_channels is not None:
        train_dataset = eeg_dataset(source_train_x[:,use_channels,:],source_train_y,source_train_s)
    else:
        train_dataset = eeg_dataset(source_train_x,source_train_y,source_train_s)
    
    if use_channels is not None:
        valid_datset = eeg_dataset(source_valid_x[:,use_channels,:],source_valid_y,source_valid_s)
    else:
        valid_datset = eeg_dataset(source_valid_x,source_valid_y,source_valid_s)
    
    return train_dataset,valid_datset,test_dataset


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
    train_valid_dataset = eeg_dataset(
        feature=torch.cat(
            [train_dataset.x, test_dataset.x],
            dim=0,
        ),
        label=torch.cat(
            [train_dataset.y, test_dataset.y],
            dim=0,
        ),
        subject_id=torch.cat(
            [train_dataset.s, test_dataset.s],
            dim=0,
        ),
    )

    return train_valid_dataset, valid_dataset, test_dataset

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
