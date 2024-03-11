import os
import numpy as np
from copy import copy
from typing import Optional, List
from nvidia.dali import pipeline_def, fn, backend
# from nvidia.dali.plugin.pytorch import DALIGenericIterator
from data.dali.iterator import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from datetime import datetime, timedelta
from data import Metadata
from torch.utils.data import DataLoader
import nvidia.dali.types as types
import math
from nvidia.dali import Pipeline
import random
from collections import defaultdict

@pipeline_def
def File2Tensor(files, random_shuffle):
    inputs = []
    for config_name, fpaths in files.items():
        if random_shuffle:
            seed=1028
            # the seed should be the same to keep the output of the readers aligning with each other.
            inputs.append(fn.readers.numpy(name=config_name, device='cpu', files=fpaths, 
                                            seed=seed, random_shuffle=True))
        else:
            inputs.append(fn.readers.numpy(name=config_name, device='cpu', files=fpaths))

    # inputs = [i.gpu() for i in inputs] 
    return (*inputs,)


def to_datetime(date_string):  
    return datetime.strptime(date_string, '%Y%m%d%H.npy')

class CropIterator:
    def __init__(self, data, timestamp, time_config, mode, crop_cnt, fold_ratio_h=1, 
                 fold_ratio_w=1, positions=None, crop_h=64, crop_w=64, h=105, w=179, 
                 margin=2, pad=4, hw_idxs=[]):
        self.mode = mode
        self.meta_datas = {
            "hrrr": {
                "single_vars": ["msl", "2t", "10u", "10v"],
                "atmos_vars": ["hgtn", "u", "v", "t", "q"],
                "atmos_levels" : [5000., 10000., 15000., 20000., 25000., 30000., 40000., 
                                  50000., 60000., 70000., 85000., 92500., 100000.]
            },
            "era5": {
                "single_vars": ["msl", "2t", "10u", "10v"],
                "atmos_vars": ["z", "u", "v", "t", "q"],
                "atmos_levels" : [5000., 10000., 15000., 20000., 25000., 30000., 40000., 
                                  50000., 60000., 70000., 85000., 92500., 100000.]
            }
        }
        self.h = h
        self.w = w
        self.timestamp = timestamp
        self.fold_ratio_h = fold_ratio_h
        self.fold_ratio_w = fold_ratio_w

        self.pad = pad
        self.crop_h, self.crop_w = crop_h, crop_w
        self.h_rand_range = list(range(pad+margin, h-pad-crop_h*fold_ratio_h-margin))
        self.w_rand_range = list(range(pad+margin, w-pad-crop_w*fold_ratio_w-margin))
        self.positions = positions

        self.idx = 0
        self.time_config = time_config
        self.data = self._rearrange(data)
        
        self.crop_cnt = crop_cnt
        self.hw_idxs = hw_idxs
        

    def __iter__(self):
        return self

    def _rearrange(self, data):
        # the key with the format 'name_input_x'
        input = []
        output = []
        data2 = {}
        
        for data_name, config in self.time_config.items():
            input = [data[f'{data_name}_input_{i}'] for i in range(config['input']-1, -1, -1)]    
            if len(input[0].shape) == 4:
                input = torch.stack(input, dim=1)
            else:
                input = torch.concat(input, dim=1)
            output = [data[f'{data_name}_output_{i}'] for i in range(config['output'])]
            if len(output[0].shape) == 4:
                output = torch.stack(output, dim=1)
            else:
                output = torch.concat(output, dim=1)
            
            data2[f"input_{data_name}"] = input
            data2[f"output_{data_name}"] = output

        return data2

    def __crop(self, data: torch.Tensor, idx_h, idx_w, scale=1):
        fold_ratio_h = self.fold_ratio_h
        fold_ratio_w = self.fold_ratio_w
        if idx_h+self.crop_h*self.fold_ratio_h > self.h:
            fold_ratio_h = (self.h-idx_h)//self.crop_h
        if idx_w+self.crop_w*self.fold_ratio_w > self.w:
            fold_ratio_w = (self.w-idx_w)//self.crop_w
        up, down, left, right = (idx_h*scale, (idx_h+self.crop_h*fold_ratio_h)*scale, 
                                     idx_w*scale, (idx_w+self.crop_w*fold_ratio_w)*scale)
            
        cropped_data = data[:, :, :, up:down, left:right]
        if self.fold_ratio_h==1 and self.fold_ratio_w==1:
            return cropped_data
        else:
            # fold into fold_ratio*fold_ratio blocks
            # (B, T, V, H, W)
            B, T, V = cropped_data.shape[:3]
            cropped_data = cropped_data.reshape(B, T, V, fold_ratio_h, self.crop_h*scale, 
                                                fold_ratio_w, self.crop_w*scale)
            
            cropped_data = cropped_data.permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, T, V, 
                                                                             self.crop_h*scale, 
                                                                             self.crop_w*scale)
            return cropped_data

    def __get_topleft(self, original_batch_size, idx_h, idx_w):
        # compose one sample positions
        fold_ratio_h, fold_ratio_w = self.fold_ratio_h, self.fold_ratio_w
        crop_h, crop_w = self.crop_h, self.crop_w
        batch_idx_h = torch.arange(fold_ratio_h)*crop_h+idx_h
        batch_idx_w = torch.arange(fold_ratio_w)*crop_w+idx_w
        batch_idx_h = batch_idx_h.reshape(1, -1, 1).repeat(original_batch_size, 1, fold_ratio_w).flatten()
        batch_idx_w = batch_idx_w.reshape(1, 1, -1).repeat(original_batch_size, fold_ratio_h, 1).flatten()
        return batch_idx_h, batch_idx_w

    def _rand_crop(self):
        if self.mode != 'test':
            idx_h = random.choice(self.h_rand_range) if len(self.h_rand_range)!=0 else 0 
            idx_w = random.choice(self.w_rand_range) if len(self.w_rand_range)!=0 else 0 
        else:
            idx_h, idx_w = self.hw_idxs[self.idx]

        batch = {}
        for k, data in self.data.items():
            batch_size = data.shape[0]
            if 'era5' in k:
                single_len = len(self.meta_datas['era5']['single_vars'])
                # atmos_len = len(self.meta_datas['era5']['atmos_vars'])
                cropped_data = self.__crop(data, idx_h, idx_w, scale=1)
                b, t = cropped_data.shape[:2]
                single = cropped_data[:, :, :single_len]
                atmos = cropped_data[:, :, single_len:].reshape(b, t, -1, 
                                                                self.crop_h+2*self.pad, 
                                                                self.crop_w+2*self.pad)
                batch[k] = {
                    'surface': single,
                    'pressure': atmos
                }
            elif 'hrrr' in k:
                scale = 10
                single_len = len(self.meta_datas['hrrr']['single_vars'])
                # atmos_len = len(self.meta_datas['hrrr']['atmos_vars'])
                cropped_data = self.__crop(data, idx_h, idx_w, scale)
                b, t = cropped_data.shape[:2]
                single = cropped_data[:, :, :single_len]
                atmos = cropped_data[:, :, single_len:].reshape(b, t, -1,
                                                                self.crop_h*scale, 
                                                                self.crop_w*scale)

                batch[k] = {
                    'surface': single,
                    'pressure': atmos
                }
            elif 'mrms' in k:
                scale = 30
                cropped_data = self.__crop(data, idx_h, idx_w, scale)
                if not torch.isnan(cropped_data).any():
                    batch[k] = cropped_data
            elif 'station' in k:
                # station data keeps the same resolution as HRRR
                scale = 10
                cropped_data = self.__crop(data, idx_h, idx_w, scale)
                batch[k] = cropped_data
            elif 'goes' in k:
                if 'goes' in self.positions and (idx_h, idx_w) in self.positions['goes']:
                    # if 
                    # station data keeps the same resolution as HRRR
                    scale = 10
                    cropped_data = self.__crop(data, idx_h, idx_w, scale)
                    batch[k] = cropped_data
        
        topleft_h, topleft_w = self.__get_topleft(batch_size, idx_h, idx_w)

        batch['meta_info'] = copy(self.meta_datas)
        if self.timestamp is not None:
            batch['meta_info']['timestamp'] = self.timestamp
        batch['meta_info']['top_left'] = (topleft_h, topleft_w)
        batch['meta_info']['height'] = self.crop_h
        batch['meta_info']['width'] = self.crop_w
        return batch

    def __next__(self):
        if self.idx < self.crop_cnt:
            d = self._rand_crop()
            self.idx += 1
            return d
        else:
            raise StopIteration


class MultiDataIterator(DALIGenericIterator):
    def __init__(self, pipelines:Pipeline, timestamps, output_map, time_config, mode,  
                 crop_cnt=3, fold_ratio=1, position_file=None, crop_h=64, crop_w=64, 
                 h=105, w=179, margin=3, pad=16, overlap=3,
                 size=-1, reader_name=None, 
                 auto_reset=False, fill_last_batch=None, dynamic_shape=False, 
                 last_batch_padded=False, last_batch_policy=LastBatchPolicy.FILL, 
                 prepare_first_batch=True):
        super().__init__(pipelines, output_map, size, reader_name, auto_reset, 
                         fill_last_batch, dynamic_shape, last_batch_padded, 
                         last_batch_policy, prepare_first_batch)
        self.mode = mode
        self.timestamps = timestamps
        self.cur_idx = 0
        # crop related properties
        self.crop_cnt = crop_cnt
        if isinstance(fold_ratio, int):
            self.fold_ratio_h = self.fold_ratio_w = fold_ratio 
        else:
            self.fold_ratio_h, self.fold_ratio_w = fold_ratio 
            

        self.positions = dict()
        if position_file is not None:
            for k, v in position_file.items():
                self.positions[k] = self._read_positions(v)

        self.crop_h = crop_h
        self.crop_w = crop_w
        self.h = h
        self.w = w
        self.margin = margin
        self.pad = pad
        self.crop_iter = None

        self.time_config = time_config
        
        self.overlap = overlap
        if self.mode == 'test':
            cropped_hsize = crop_h*self.fold_ratio_h
            cropped_wsize = crop_w*self.fold_ratio_w
            self.hw_idxs = []
            for hi in range(0, h-crop_h+1, cropped_hsize-2*self.overlap):
                for wj in range(0, w-crop_h+1, cropped_wsize-2*self.overlap):
                    self.hw_idxs.append((hi, wj))
            self.crop_cnt = len(self.hw_idxs)
            print(self.hw_idxs)
        else:
            self.hw_idxs = []
    
    def _read_positions(self, position_file):
        positions = []
        with open(position_file, 'r') as fp:
            lines = fp.readlines()
            for l in lines:
                lsplit = l.split(' ')
                i, j = int(lsplit[0]), int(lsplit[1])
                positions.append((i, j))
        positions = set(positions)
        return positions
        
    def __len__(self):
        return super().__len__() * self.crop_cnt
    
    def reset(self):
        # reset
        self.cur_idx = 0
        super().reset()

    def __next__(self):
        if self.crop_iter is None:
            data = super().__next__()
            self.crop_iter = CropIterator(data[0], self.timestamps[self.cur_idx:(self.cur_idx+self.batch_size)], 
                                          self.time_config, self.mode, self.crop_cnt, self.fold_ratio_h, self.fold_ratio_w, 
                                          self.positions, self.crop_h, self.crop_w, self.h, self.w, 
                                          self.margin, self.pad, self.hw_idxs)
            self.cur_idx += self.batch_size
        
        try:
            return next(self.crop_iter)
        except StopIteration:
            self.crop_iter = None
            # call it again to see if there is more data.
            return next(self)

def check_valid(datetime, datetime_set, input_horizon, gap, output_horizon):
    for i in range(input_horizon):
        if datetime - timedelta(hours=i) not in datetime_set:
            return False
    for i in range(output_horizon):
        if datetime + timedelta(hours=gap+i) not in datetime_set:
            return False
    return True
    

class DALIDataModule(LightningDataModule):
    def __init__(self, 
                 data_dir, 
                 batch_size = 2,
                 time_config = {
                                'hrrr': {'input': 2, 'output': 1, 'gap': 1}, 
                                'era5': {'input': 2, 'output': 1, 'gap': 1},
                                'mrms': {'input': 2, 'output': 1, 'gap': 1}
                                },
                 num_threads = 8,
                 mount_paths = ['/nfs1'],
                 position_file = None,
                 train_ratio = None,
                 train_end_datetime = '2020060100',
                 valid_size = 7*24,
                 crop_cnt=3, fold_ratio=1, crop_h=64, crop_w=64, 
                 h=106, w=180, margin=4, pad=16,
                 prefetch_queue_depth=1, overlap=3,
                 ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.prepare()
        self.setup()

    def _fill_mount_path(self, times, extension_name='npy'):
        fcnt_per_paths = len(times)//len(self.hparams.mount_paths) + 1
        file_mount_paths = self.hparams.mount_paths*fcnt_per_paths
        files = defaultdict(list)
        for t, m in zip(times, file_mount_paths):
            for data_name, config in self.hparams.time_config.items():
                data_dir = self.hparams.data_dir[data_name]
                for i in range(config['input']):
                    input_t = t - timedelta(hours=i)
                    fname = input_t.strftime(f"%Y%m%d%H.{extension_name}")
                    files[f'{data_name}_input_{i}'].append(os.path.join(m, data_dir, 
                                                                        fname))
                for j in range(config['output']):
                    output_t = t + timedelta(hours=j+config['gap'])
                    fname = output_t.strftime(f"%Y%m%d%H.{extension_name}")
                    files[f'{data_name}_output_{j}'].append(os.path.join(m, data_dir, 
                                                                        fname))
                    
        timestamps = copy(times)
        total_data_size = len(times)
        return files, timestamps, total_data_size

    def prepare(self):
        times = None
        extension_name = 'npy' 
        data_times = {}
        base_path = '/nfs' if 'nfs' in self.hparams.mount_paths[0] else self.hparams.mount_paths[0]
        for data_name, data_dir in self.hparams.data_dir.items():
            data_times[data_name] = set([datetime.strptime(f, "%Y%m%d%H.npy") for f 
                                in os.listdir(os.path.join(base_path, data_dir)) 
                                if f.endswith(f'.{extension_name}')])
            if times is None:
                times = copy(data_times[data_name])
            else:
                # get the joint space.
                times = data_times[data_name] & times
        # Get those valid datetime samples.
        valid_times = []
        for t in times:
            validity = True
            for data_name, config in self.hparams.time_config.items():
                if not check_valid(t, data_times[data_name], input_horizon=config['input'], 
                            gap=config['gap'], output_horizon=config['output']):
                    validity = False
                    break
            if validity:
                valid_times.append(t)
        valid_times = sorted(valid_times)
        # valid_times = valid_times[:10]

        if self.hparams.train_ratio is not None:
            # train_ration is the amount of training data.
            train_size = int(len(valid_times)*self.hparams.train_ratio)
        elif self.hparams.train_end_datetime is not None:
            end_datetime = datetime.strptime(str(self.hparams.train_end_datetime), "%Y%m%d%H")
            for i, d in enumerate(valid_times):
                if d >= end_datetime:
                    break
            train_size = i
        else:
            train_size = 0.75

        valid_size = self.hparams.valid_size
        train_times, val_times, test_times = (valid_times[:train_size], 
                                   valid_times[train_size:train_size+valid_size], 
                                   valid_times[train_size+valid_size:])

        self.train_files, self.train_timestamps, self.train_data_size = self._fill_mount_path(train_times)
        self.val_files, self.val_timestamps, self.val_data_size = self._fill_mount_path(val_times)
        self.test_files, self.test_timestamps, self.test_data_size = self._fill_mount_path(test_times)

        
    def _sharding(self, shard_id, num_shard, files, timestamps, data_size):
        sample_per_shard = data_size // num_shard
        start_idx, end_idx = shard_id*sample_per_shard, (shard_id+1)*sample_per_shard

        local_paths = {k: v[start_idx: end_idx] for k, v in files.items()}
        local_timestampes = timestamps[start_idx: end_idx]
        
        local_file_idxs = list(range(end_idx-start_idx))
        return local_paths, local_timestampes, local_file_idxs
        
        
    def setup(self, stage=None):
        if self.trainer is None:
            shard_id = 0
            num_shard = 1
            self.device_id = 0
        else:
            shard_id = self.trainer.global_rank
            num_shard = self.trainer.world_size
            self.device_id = self.trainer.local_rank
        train_local_paths, train_local_time_stamps, train_local_file_idxs = self._sharding(shard_id, 
                                                                                            num_shard,
                                                                                            self.train_files,
                                                                                            self.train_timestamps,
                                                                                            self.train_data_size)
        
        val_local_paths, val_local_time_stamps, val_local_file_idxs = self._sharding(shard_id, 
                                                                                    num_shard,
                                                                                    self.val_files,
                                                                                    self.val_timestamps,
                                                                                    self.val_data_size)
        
        test_local_paths, test_local_time_stamps, test_local_file_idxs = self._sharding(shard_id, 
                                                                                    num_shard,
                                                                                    self.test_files,
                                                                                    self.test_timestamps,
                                                                                    self.test_data_size)
        
        self.local_paths = {
            'train': train_local_paths,
            'val': val_local_paths,
            'test': test_local_paths
        }
        self.local_timestampes = {
            'train': train_local_time_stamps,
            'val': val_local_time_stamps,
            'test': test_local_time_stamps
        }
        self.local_file_idxs = {
            'train': train_local_file_idxs,
            'val': val_local_file_idxs,
            'test': test_local_file_idxs
        }
        
    def _get_dataloader(self, mode):
        if mode == 'train':
            random.shuffle(self.local_file_idxs[mode])
            for k in self.local_paths[mode]:
                self.local_paths[mode][k] = [self.local_paths[mode][k][i] 
                                            for i in self.local_file_idxs[mode]]
            self.local_timestampes[mode] = [self.local_timestampes[mode][i] 
                                            for i in self.local_file_idxs[mode]]
            # random_shuffle = True
            # TODO:Shutdown shuffle termperarily, Unfixed bug.
            random_shuffle = False
        else:
            random_shuffle = False

        npy_pipe = File2Tensor(batch_size=self.hparams.batch_size, 
                              num_threads=self.hparams.num_threads,
                              device_id=self.device_id,
                              files=self.local_paths[mode],
                              set_affinity=True,
                              prefetch_queue_depth=self.hparams.prefetch_queue_depth, 
                              random_shuffle=random_shuffle)
        # npy_pipe.build()

        data_names = list(self.local_paths[mode].keys())
        dataloader = MultiDataIterator(npy_pipe, 
                                            timestamps=self.local_timestampes[mode],
                                            output_map=data_names, 
                                            time_config=self.hparams.time_config,
                                            mode=mode,
                                            crop_cnt=self.hparams.crop_cnt, 
                                            fold_ratio=self.hparams.fold_ratio,
                                            position_file=self.hparams.position_file,
                                            crop_h=self.hparams.crop_h, 
                                            crop_w=self.hparams.crop_w, 
                                            h=self.hparams.h, 
                                            w=self.hparams.w, 
                                            margin=self.hparams.margin, 
                                            pad=self.hparams.pad,
                                            overlap=self.hparams.overlap,
                                            reader_name=data_names[0], 
                                            last_batch_policy=LastBatchPolicy.DROP, 
                                            prepare_first_batch=True)

        return dataloader

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        backend.ReleaseUnusedMemory()
        return self._get_dataloader('train')

    def val_dataloader(self) -> EVAL_DATALOADERS:
        backend.ReleaseUnusedMemory()
        return self._get_dataloader('val')

    def test_dataloader(self) -> EVAL_DATALOADERS:
        backend.ReleaseUnusedMemory()
        return self._get_dataloader('test')
