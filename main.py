import torch.nn as nn
import torchshow as ts
from data.dali.module import DALIDataModule
from tqdm import tqdm
import torch
import sys
import numpy as np
import pickle
import os
import random
import warnings
from loguru import logger
import atexit
import signal
import argparse
from SRFormer import SRFormer, FlexibleTqdm, LearningCurve

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

warnings.filterwarnings('ignore')
sys.path.append('/home/whx/pl_SRFormer')

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--seed', default=2024, type=int,
                    help='random seed')
parser.add_argument('--model', default="SRFormer", type=str,
                    help='model name')
parser.add_argument('--bs', default=8, type=int,
                    help='batch_size')
parser.add_argument('--epochs', default=10, type=int,
                    help='Epochs')
parser.add_argument('--gpu', default=0, type=int,
                    help='cuda:')
args = parser.parse_args()

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

def cleanup_log_file():
    """
    清理日志文件的函数。这将在程序退出时被调用。
    """
    try:
        os.remove(log_file_path)
        print(f"Log file {log_file_path} has been removed.")
    except OSError as e:
        print(f"Error: {e.strerror} - {e.filename}")

log_file_path = os.path.join('/home/whx/pl_SRFormer/log', '{}_epo_{}_lr_{}_bs_{}.log'.format(args.model,
        args.epochs, args.lr, args.bs))
logger.add(log_file_path, rotation="50 MB", level="INFO") #每50MB重新换一个log文件，并且将logger输出的信息保存在磁盘上
atexit.register(cleanup_log_file)
def handle_kill_signal(signum, frame):
    cleanup_log_file()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_kill_signal)
signal.signal(signal.SIGINT, handle_kill_signal)

logger.info(args)

VAR_LIST = ['msl', '2t', '10u', '10v', 
            'hgtn_5000', 'hgtn_10000', 'hgtn_15000', 'hgtn_20000', 'hgtn_25000', 'hgtn_30000', 'hgtn_40000', 'hgtn_50000', 'hgtn_60000', 'hgtn_70000', 'hgtn_85000', 'hgtn_92500', 'hgtn_100000', 
            'u_5000', 'u_10000', 'u_15000', 'u_20000', 'u_25000', 'u_30000', 'u_40000', 'u_50000', 'u_60000', 'u_70000', 'u_85000', 'u_92500', 'u_100000', 
            'v_5000', 'v_10000', 'v_15000', 'v_20000', 'v_25000', 'v_30000', 'v_40000', 'v_50000', 'v_60000', 'v_70000', 'v_85000', 'v_92500', 'v_100000', 
            't_5000', 't_10000', 't_15000', 't_20000', 't_25000', 't_30000', 't_40000', 't_50000', 't_60000', 't_70000', 't_85000', 't_92500', 't_100000', 
            'q_5000', 'q_10000', 'q_15000', 'q_20000', 'q_25000', 'q_30000', 'q_40000', 'q_50000', 'q_60000', 'q_70000', 'q_85000', 'q_92500', 'q_100000'
]

data_dir = {
        "hrrr": 'hrrr/hourly2_fixed_TMP_L103',
        "era5": 'era5_us_npy/npy',
    }
input_len, output_len = 1, 1
time_config = {
        'hrrr': {'input': input_len, 'output': output_len, 'gap': 1}, 
        'era5': {'input': input_len, 'output': output_len, 'gap': 1}, 
    }
crop_h= 32
crop_w= 32
crop_cnt= 1
batch_size= args.bs
pad= 0
margin= 0
h= 105
w= 179
mount_paths = ['/blob/kmsw0westus3/kms1']

data_module = DALIDataModule(data_dir=data_dir, batch_size=batch_size, num_threads=8, 
                            time_config=time_config, mount_paths=mount_paths, 
                            crop_cnt=crop_cnt, crop_h=crop_h, crop_w=crop_w, h=h, w=w, 
                            margin=margin, pad=pad, train_end_datetime='2020123100')

data_module.prepare()
data_module.setup()
train_dataloader = data_module.train_dataloader()
val_dataloader = data_module.val_dataloader()

ALL_STDS = torch.Tensor([6.8789e+02, 1.1408e+01, 3.4087e+00, 3.8128e+00, 1.9669e+02, 2.2751e+02,
2.8561e+02, 3.0650e+02, 2.9603e+02, 2.7141e+02, 2.1778e+02, 1.7208e+02,
1.3423e+02, 1.0113e+02, 6.2827e+01, 5.5131e+01, 5.6151e+01, 9.4211e+00,
1.0859e+01, 1.3950e+01, 1.7115e+01, 1.7562e+01, 1.6353e+01, 1.3276e+01,
1.0801e+01, 8.9426e+00, 7.5022e+00, 6.1020e+00, 5.4538e+00, 3.7264e+00,
4.3335e+00, 7.7346e+00, 1.2055e+01, 1.6379e+01, 1.7496e+01, 1.6438e+01,
1.3244e+01, 1.0723e+01, 8.8519e+00, 7.4927e+00, 6.6357e+00, 6.3794e+00,
4.4868e+00, 4.5961e+00, 6.9949e+00, 5.8210e+00, 4.5001e+00, 4.9734e+00,
6.2033e+00, 7.2265e+00, 7.4370e+00, 7.7510e+00, 8.4697e+00, 9.9353e+00,
1.0570e+01, 1.0831e+01, 1.0117e-06, 2.1395e-06, 4.7380e-06, 1.8255e-05,
6.3045e-05, 1.5294e-04, 4.8075e-04, 9.9271e-04, 1.6142e-03, 2.2498e-03,
3.5426e-03, 4.3892e-03, 5.0458e-03])

ALL_MEANS = torch.Tensor([1.0154e+05,  2.8764e+02,  3.6334e-01, -2.1942e-01,  2.0653e+04,
1.6401e+04,  1.3901e+04,  1.2086e+04,  1.0639e+04,  9.4158e+03,
7.3890e+03,  5.7308e+03,  4.3219e+03,  3.0932e+03,  1.4998e+03,
7.9164e+02,  1.2938e+02,  1.9525e+00,  1.3733e+01,  2.1062e+01,
2.3357e+01,  2.1682e+01,  1.9152e+01,  1.4822e+01,  1.1510e+01,
8.7861e+00,  6.3429e+00,  2.7486e+00,  1.2042e+00,  3.4059e-01,
-1.9104e-01,  1.5638e-01,  3.5123e-01,  4.5588e-01,  4.4878e-01,
3.9698e-01,  2.3847e-01,  1.6056e-01,  1.3170e-01,  1.8216e-01,
3.9488e-01,  3.1365e-01, -2.5557e-01,  2.1192e+02,  2.0901e+02,
2.1325e+02,  2.1837e+02,  2.2547e+02,  2.3356e+02,  2.4815e+02,
2.5946e+02,  2.6831e+02,  2.7558e+02,  2.8359e+02,  2.8666e+02,
2.9031e+02,  2.8537e-06,  3.0846e-06,  6.6055e-06,  2.4169e-05,
7.3131e-05,  1.6582e-04,  4.9159e-04,  1.0437e-03,  1.8908e-03,
2.9435e-03,  5.3577e-03,  7.2076e-03,  8.8479e-03])

device = torch.device("cuda:%d" % args.gpu)
torch.cuda.set_device(args.gpu)

model = SRFormer(img_size=32, upsampler = 'pixelshuffledirect',in_chans=69,
                 window_size=2, embed_dim=768, depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6], 
                 num_heads=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4], upscale=10).to(device)

model_dir = '/home/whx/pl_SRFormer/ckpt/' + args.model
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
epochs = args.epochs

logger.info('Start to Training!')

trainer_params = {
    "devices": [0,1,2,3],
    "accelerator":"gpu",
    "precision":"16",
    "max_epochs": epochs,  # 1000
    "logger": False,  # TensorBoardLogger
    "callbacks": [
        # pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        # pl.callbacks.ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min"),
        pl.callbacks.ModelCheckpoint(every_n_train_steps=1000, save_top_k=-1),
        FlexibleTqdm(data_module.train_data_size // batch_size, column_width=12), # 注意设置progress_bar_refresh_rate=0，取消自带的进度条
        LearningCurve(figsize=(12, 4), names=("loss", "mse")),
    ],  # None
}
trainer = pl.Trainer(**trainer_params)

trainer.fit(model, train_dataloader, val_dataloader)
