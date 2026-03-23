# The code is borrowed from Beat Transformer (https://github.com/zhaojw1998/Beat-Transformer/)
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import time
import numpy as np
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
import madmom
import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from optimizer import Lookahead
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils import AverageMeter, epoch_time, infer_beat_with_DBN, infer_downbeat_with_DBN
from spectrogram_dataset import audioDataset
from torch.nn.utils.rnn import pad_sequence
from model import MambaBeatTracker
#from focal_loss import BCEWithLogitsFocalLoss
from torch.cuda.amp import GradScaler
import gc
import warnings

warnings.filterwarnings('ignore')

FOLD = int(sys.argv[1])
GPU = int(sys.argv[2])
PROJECT_NAME = 'Beat-drumMamba'
torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.deterministic = False

###############################################################################
# Load config
###############################################################################
SAMPLE_SIZE = 512
FPS = 44100 / 1024
NUM_FOLDS = 8

# Mamba model configuration
model_config = {
    'input_features': 256, 
    'input_channels': 2, 
    'fusion_type': 'concat',
    'dmodel': 256,  
    'depth': 9,  
    'patch_size': 8,  
    'num_classes': 2,  
    'drop_rate': 0.1,
    'drop_path_rate': 0.1,
    'final_pool_type': 'all',  
    'use_cls_token': False,
    'bimamba_type': 'v2'
}

# training
DEVICE = f'cuda:{GPU}'
TRAIN_BATCH_SIZE = 8
LEARNING_RATE = 1.5e-3  
DECAY = 0.99995
N_EPOCH = 40
CLIP = .5

# 梯度累积配置
ACCUMULATION_STEPS = 4  

# directories
DATASET_PATH = '/media/oem/dataset/Li_data/beat_spectrogram_data1.npz'
ANNOTATION_PATH = '/media/oem/dataset/Li_data/beat_annotation1.npz'
DATA_TO_LOAD = ['ballroom', 'hainsworth', 'smc', 'carnetic', 'gtzan', 'harmonix']
TEST_ONLY = ['gtzan']
SAVE_PATH = f'../save/train_log/{str(GPU).zfill(2)}_{PROJECT_NAME}'

print(f'\nProject initialized: {PROJECT_NAME}\n', flush=True)
print(f'\nFold {FOLD}')

###############################################################################
# Initialize fold
###############################################################################
project_path = os.path.join(SAVE_PATH, f'Fold_{FOLD}')

MODEL_PATH = os.path.join(project_path, 'model')
LOG_PATH = os.path.join(project_path, 'log')

print(MODEL_PATH)
print(LOG_PATH)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

loss_writer = SummaryWriter(os.path.join(LOG_PATH, 'loss'))
beat_ll_writer = SummaryWriter(os.path.join(LOG_PATH, 'beat_likelihood'))
beat_pr_writer = SummaryWriter(os.path.join(LOG_PATH, 'beat_precision'))
beat_DBN_writer = SummaryWriter(os.path.join(LOG_PATH, 'beat_DBN_acc'))



# 内存优化函数
###############################################################################
def clear_memory():
    """清理GPU内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_usage():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        cached = torch.cuda.memory_reserved() / 1024 ** 3
        return f"Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
    return "CUDA not available"


###############################################################################
# Model initialization
###############################################################################
# scaler = GradScaler()

# 创建MambaBeatTracker模型
model = MambaBeatTracker(**model_config,return_tempo=True)
model.to(DEVICE)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# 打印模型结构
print("Model architecture:")
print(model)

###############################################################################
# Load data
###############################################################################
dataset = audioDataset(data_to_load=DATA_TO_LOAD,
                       test_only_data=TEST_ONLY,
                       data_path=DATASET_PATH,
                       annotation_path=ANNOTATION_PATH,
                       fps=FPS,
                       sample_size=SAMPLE_SIZE,
                       num_folds=NUM_FOLDS)


def collate_fn_pad(batch):
    """填充到相同长度的collate函数"""
    dataset_keys, data_list, beat_gt_list, downbeat_gt_list, tempo_gt_list = zip(*batch)

    # 转换为张量
    data_tensors = [torch.as_tensor(data, dtype=torch.float32) for data in data_list]
    beat_gt_tensors = [torch.as_tensor(beat_gt, dtype=torch.float32) for beat_gt in beat_gt_list]
    downbeat_gt_tensors = [torch.as_tensor(downbeat_gt, dtype=torch.float32) for downbeat_gt in downbeat_gt_list]
    tempo_gt_tensors = [torch.as_tensor(tempo_gt, dtype=torch.float32) for tempo_gt in tempo_gt_list]

    # 填充序列
    data_padded = pad_sequence(data_tensors, batch_first=True, padding_value=0.0)   #(8,1937,128)
    beat_gt_padded = pad_sequence(beat_gt_tensors, batch_first=True, padding_value=-1.0)  #(8,1937)
    downbeat_gt_padded = pad_sequence(downbeat_gt_tensors, batch_first=True, padding_value=-1.0)  #(8,1937)
    tempo_gt_stacked = torch.stack(tempo_gt_tensors)  #(8,1937)   

    return (list(dataset_keys), data_padded, beat_gt_padded, downbeat_gt_padded, tempo_gt_stacked)


train_set, val_set, test_set = dataset.get_fold(fold=FOLD)
train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad)
val_loader = DataLoader(val_set,1, shuffle=False)

###############################################################################
# Optimizer and Criterion
###############################################################################
optimizer = optim.RAdam(model.parameters(), lr=LEARNING_RATE)
optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)
# optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05, betas=(0.9,0.95))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,  factor=.2, patience=2, verbose=True,
                                                       threshold=1e-3, min_lr=1e-7)

# 损失函数 - 注意权重类型修正
loss_func = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.FloatTensor([1, 1]).to(DEVICE))
loss_tempo = nn.BCEWithLogitsLoss(reduction='none')

beat_tracker = madmom.features.beats.DBNBeatTrackingProcessor(min_bpm=55.0, max_bpm=215.0, fps=FPS, transition_lambda=2,
                                                              observation_lambda=8, threshold=0.05, online=True)
downbeat_tracker = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], min_bpm=55.0,
                                                                          max_bpm=215.0, fps=FPS, transition_lambda=10,
                                                                          observation_lambda=12, threshold=0.05,
                                                                          online=True)


###############################################################################
# Training and Evaluation Functions
###############################################################################
def train(model, train_loader, optimizer, scheduler, loss_func, loss_tempo, clip, epoch, device):
    print('training ...', flush=True)
    num_batch = len(train_loader)
    loss_meter_b = AverageMeter()
    loss_meter_t = AverageMeter()
    beat_meter = AverageMeter()
    beat_DBN_meter = AverageMeter()
    nan_count = []
    model.train()

    for idx, (dataset_key, data, beat_gt, downbeat_gt, tempo_gt) in tqdm(enumerate(train_loader),
                                                                         total=len(train_loader)):
        try:
          
            data = data.float().to(device)
            beat_gt = beat_gt.to(device)
            downbeat_gt = downbeat_gt.to(device)
            gt = torch.cat([beat_gt.unsqueeze(-1), downbeat_gt.unsqueeze(-1)], dim=-1).float().to(device)  # (batch, T', 2)
            tempo_gt = tempo_gt.to(device)

            optimizer.zero_grad()  

            pred, tempo = model(data, return_tempo=True)

            min_len = min(pred.shape[1], gt.shape[1])
            pred = pred[:, :min_len, :]
            valid_gt = gt[:, :min_len, :]

            mask = (valid_gt !=-1 ).float() 
            targets = torch.where(valid_gt == -1, torch.zeros_like(valid_gt), valid_gt)

            loss = loss_func(pred,targets)

            loss = (mask * loss).mean(dim=(0, 1)).sum()

            # 计算tempo损失
            valid_tempo_gt = tempo_gt.clone()
            valid_tempo_gt[tempo_gt == -1] = 0
            loss_t = loss_tempo(torch.softmax(tempo, dim=-1), valid_tempo_gt)
            weight_t = (1 - torch.as_tensor(tempo_gt == -1, dtype=torch.int32)).to(device)
            loss_t = (weight_t * loss_t).mean()

            loss_meter_t.update('train/loss', loss_t.item())
            loss_meter_b.update('train/loss', loss.item())
            if ((dataset_key[0] == 'musicnet') and (-1 in tempo_gt)):
                loss =  loss * 0 
            loss=loss+loss_t
            
            if torch.isnan(loss):
                nan_count.append(str(dataset_key)+'\n')
                with open('/media/oem/LLLi/BeatMamba-main/nancount.txt', 'w') as f:
                    f.writelines(nan_count)
 
            loss.backward()
            # 梯度累积
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()


        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"CUDA OOM at batch {idx}, skipping batch", flush=True)
                clear_memory()
                continue
            else:
                raise e
            
        loss_writer.add_scalar('train/loss_beat', loss_meter_b.avg['train/loss'], epoch * num_batch + idx)
        loss_writer.add_scalar('train/loss_tempo', loss_meter_t.avg['train/loss'], epoch * num_batch + idx)
        loss_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch * num_batch + idx)

    return loss_meter_b, loss_meter_t, beat_meter, beat_DBN_meter


def evaluate(model, val_loader, loss_func, loss_tempo, epoch, device):
    print('validating ...', flush=True)
    num_batch = len(val_loader)
    loss_meter_b = AverageMeter()
    loss_meter_t = AverageMeter()
    beat_meter = AverageMeter()
    beat_DBN_meter = AverageMeter()


    model.eval()

    with torch.no_grad():
        for idx, (dataset_key, data, beat_gt, downbeat_gt, tempo_gt) in tqdm(enumerate(val_loader), total=num_batch):
            try:

                data = data.float().to(device)
                # annotation
                beat_gt = beat_gt.to(device)
                downbeat_gt = downbeat_gt.to(device)
                gt = torch.cat([beat_gt.unsqueeze(-1), downbeat_gt.unsqueeze(-1)], dim=-1).float().to(
                    device)  # (batch, T', 2)
                tempo_gt = tempo_gt.float().to(device)

                pred, tempo = model(data, return_tempo=True)
                min_len = min(pred.shape[1], gt.shape[1])
                pred = pred[:, :min_len, :]
                valid_gt = gt[:, :min_len, :]
                mask = (valid_gt != -1).float()
                targets = torch.where(valid_gt == -1,torch.zeros_like(valid_gt),valid_gt)
  
                loss = loss_func(pred, targets)
                loss = (mask * loss).mean(dim=(0, 1)).sum()

                # 计算tempo损失
                valid_tempo_gt = tempo_gt.clone()
                valid_tempo_gt[tempo_gt == -1] = 0
                loss_t = loss_tempo(torch.softmax(tempo, dim=-1), valid_tempo_gt)
                weight = (1 - torch.as_tensor(tempo_gt == -1, dtype=torch.int32)).to(device)
                loss_t = (weight * loss_t).mean()

            except RuntimeError as e:
                print(f"Error in validation batch {idx}: {e}")
                continue

            # 记录损失
            if not dataset_key[0] == 'gtzan':
                loss_meter_b.update('val/loss', loss.item())
            else:
                loss_meter_b.update('val/loss_nontrain', loss.item())

            if not dataset_key[0] == 'gtzan':
                loss_meter_t.update('val/loss', loss_t.item())
            else:
                loss_meter_t.update('val/loss_nontrain', loss_t.item())

            # 合并评估所有样本
            min_len2 = min(pred.shape[1], beat_gt.shape[1], pred.shape[1], downbeat_gt.shape[1])
            pred_beat = pred[:, :min_len2, 0]           # (B, min_len)
            beat_gt_batch = beat_gt[:, :min_len2]       # (B, min_len)


            # Beat DBN评估
            beat_acc_DBN = infer_beat_with_DBN(pred_beat, beat_gt_batch, beat_tracker, FPS)
            for key in beat_acc_DBN:
                beat_DBN_meter.update(f'val-{dataset_key[0]}/{key}', beat_acc_DBN[key])


     # 记录验证损失
    if not dataset_key[0] == 'gtzan':
        loss_writer.add_scalar('val/loss_beat', loss_meter_b.avg['val/loss'], epoch)
        loss_writer.add_scalar('val/loss_tempo', loss_meter_t.avg['val/loss'], epoch)
    else:
        loss_writer.add_scalar('val/loss_beat_nontrain', loss_meter_b.avg['val/loss_nontrain'], epoch)
        loss_writer.add_scalar('val/loss_tempo_nontrain', loss_meter_t.avg['val/loss_nontrain'], epoch)

    # 记录其他指标
    for key in beat_DBN_meter.avg.keys():
        if 'val' in key:
            beat_DBN_writer.add_scalar(key, beat_DBN_meter.avg[key], epoch)

    return loss_meter_b, loss_meter_t, beat_meter, beat_DBN_meter


###############################################################################
# Main Training Loop
###############################################################################
for epoch in range(N_EPOCH):
    print(f'Start Epoch: {epoch + 1:02}', flush=True)
    start_time = time.time()

    # 训练
    _, _, _, _,  = train(model, train_loader, optimizer, scheduler, loss_func, loss_tempo, CLIP, epoch, DEVICE)

    # 验证
    loss_meter_b, loss_meter_t, beat_meter, beat_DBN_meter = evaluate(model, val_loader, loss_func, loss_tempo, epoch, DEVICE)
    scheduler.step(loss_meter_b.avg['val/loss'] + loss_meter_t.avg['val/loss'])

    # 保存模型
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'model_config': model_config,  
                }, os.path.join(MODEL_PATH, 'mamba_param_' + str(epoch).zfill(3) + '.pt'))

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s', flush=True)

    print('val beat loss:', loss_meter_b.avg['val/loss'], flush=True)
    print('val tempo loss:', loss_meter_t.avg['val/loss'], flush=True)
    #print('beat accuracy:', [(key.split('/')[-1], beat_meter.avg[key]) for key in beat_meter.avg.keys() if 'val' in key], flush=True)
    print('beat accuracy with DBN:', [(key.split('/')[-1], beat_DBN_meter.avg[key]) for key in beat_DBN_meter.avg.keys() if 'val' in key], flush=True)


    print('\n')

print("Training completed!")
