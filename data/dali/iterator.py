'''Modified version of nvidia.dali.plugin.torch.DALIGenericIterator
'''
from nvidia.dali.plugin.base_iterator import _DaliBaseIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.backend import TensorGPU, TensorListGPU
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali import types
from nvidia.dali.plugin.base_iterator import _DaliBaseIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import to_torch_type, feed_ndarray
import torch
import torch.utils.dlpack as torch_dlpack
import ctypes
import numpy as np

def decode_mrms(tensor:torch.LongTensor):
    T, H, W = 6, 3177, 5397 # this is constants

    a = (tensor >> 60) & 0xF  
    b = (tensor >> 46) & 0x3FFF  
    c = (tensor >> 32) & 0x3FFF  
    d_int64 = tensor & 0xFFFFFFFF  
  
    # Convert int32 tensor to float32 tensor  
    d = d_int64.to(torch.int32).view(torch.float32)  
    # result = torch.zeros(size=(T, H, W), dtype=torch.float32, device=tensor.device)
    # result[a, b, c] = d
    return a, b, c, d

def decode_station(tensor:torch.FloatTensor):
    idx_h = tensor[0]
    idx = (idx_h >> 16) & 0xFFFF
    h = idx_h & 0xFFFF
    w_i = tensor[1]
    w = (w_i >> 16) & 0xFFFF
    # i is the index of attribute
    i = w_i & 0xFFFF
    # r is the attribute value
    r = tensor[2].view(torch.float32)
    return idx, h, w, i, r
    
    '''
    mean = station_mean.to(tensor.device).reshape(1, -1, 1, 1)
    std = station_std.to(tensor.device).reshape(1, -1, 1, 1)
    T, H, W = 12, 1059, 1799
    # T, D, H, W
    result = mean.repeat(T, 1, H, W)
    result[idx, i, h, w] = r
    print(result.shape)
    result = (result - mean)/std
    return result
    '''

class DALIGenericIterator(_DaliBaseIterator):
    def __init__(self,
                 pipelines,
                 output_map,
                 size=-1,
                 reader_name=None,
                 auto_reset=False,
                 fill_last_batch=None,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 last_batch_policy=LastBatchPolicy.FILL,
                 prepare_first_batch=True):

        # check the assert first as _DaliBaseIterator would run the prefetch
        assert len(set(output_map)) == len(output_map), "output_map names should be distinct"
        self._output_categories = set(output_map)
        self.output_map = output_map

        _DaliBaseIterator.__init__(self,
                                   pipelines,
                                   size,
                                   reader_name,
                                   auto_reset,
                                   fill_last_batch,
                                   last_batch_padded,
                                   last_batch_policy,
                                   prepare_first_batch=prepare_first_batch)

        self._first_batch = None
        if self._prepare_first_batch:
            try:
                self._first_batch = DALIGenericIterator.__next__(self)
                # call to `next` sets _ever_consumed to True but if we are just calling it from
                # here we should set if to False again
                self._ever_consumed = False
            except StopIteration:
                assert False, "It seems that there is no data in the pipeline. This may happen " \
                       "if `last_batch_policy` is set to PARTIAL and the requested batch size is " \
                       "greater than the shard size."

    def __next__(self):
        self._ever_consumed = True
        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch

        # Gather outputs
        outputs = self._get_outputs()

        data_batches = [None for i in range(self._num_gpus)]
        for i in range(self._num_gpus):
            dev_id = self._pipes[i].device_id
            # initialize dict for all output categories
            category_outputs = dict()
            # segregate outputs into categories
            for j, out in enumerate(outputs[i]):
                category_outputs[self.output_map[j]] = out

            # Change DALI TensorLists into Tensors
            category_tensors = dict()
            category_shapes = dict()
            for category, out in category_outputs.items():
                if 'mrms' in category or 'station' in category:
                    category_tensors[category] = out
                    category_shapes[category] = [c.shape() for c in out]
                else:
                    category_tensors[category] = out.as_tensor()
                    category_shapes[category] = category_tensors[category].shape()

            category_torch_type = dict()
            category_device = dict()
            torch_gpu_device = None
            torch_cpu_device = torch.device('cpu')
            # check category and device
            for category in self._output_categories:
                if 'mrms' in category or 'station' in category:
                    category_torch_type[category] = to_torch_type[category_tensors[category][0].dtype]
                else:
                    category_torch_type[category] = to_torch_type[category_tensors[category].dtype]

                if isinstance(category_tensors[category], (TensorGPU, TensorListGPU)):
                    if not torch_gpu_device:
                        torch_gpu_device = torch.device('cuda', dev_id)
                    category_device[category] = torch_gpu_device
                else:
                    category_device[category] = torch_cpu_device

            pyt_tensors = dict()
            for category in self._output_categories:
                if 'mrms' in category or 'station' in category:
                    pyt_tensors[category] = [torch.empty(c, 
                                                         dtype=category_torch_type[category], 
                                                         device=category_device[category]) 
                                             for c in category_shapes[category]]
                else:
                    pyt_tensors[category] = torch.empty(category_shapes[category],
                                                        dtype=category_torch_type[category],
                                                        device=category_device[category])
                

            data_batches[i] = pyt_tensors

            # Copy data from DALI Tensors to torch tensors
            for category, tensor in category_tensors.items():
                if 'mrms' in category or 'station' in category:
                    cat_tensors = []
                    for i, t in enumerate(tensor):
                        if isinstance(tensor, (TensorGPU, TensorListGPU)):
                            stream = torch.cuda.current_stream(device=pyt_tensors[category][i].device)
                            feed_ndarray(t, pyt_tensors[category][i], cuda_stream=stream)
                        else:
                            feed_ndarray(t, pyt_tensors[category][i])
                        
                        if 'mrms' in category:
                            decoded = decode_mrms(pyt_tensors[category][i])
                        else:
                            decoded = decode_station(pyt_tensors[category][i])
                        # decode from float64 to a, b, c, d.
                        cat_tensors.append(decoded)
                    # pyt_tensors[category] = torch.stack(cat_tensors, dim=0)
                    pyt_tensors[category] = cat_tensors
                elif isinstance(tensor, (TensorGPU, TensorListGPU)):
                    # Using same cuda_stream used by torch.zeros to set the memory
                    stream = torch.cuda.current_stream(device=pyt_tensors[category].device)
                    feed_ndarray(tensor, pyt_tensors[category], cuda_stream=stream)
                else:
                    feed_ndarray(tensor, pyt_tensors[category])

        self._schedule_runs()

        self._advance_and_check_drop_last()

        if self._reader_name:
            if_drop, left = self._remove_padded()
            if np.any(if_drop):
                output = []
                for batch, to_copy in zip(data_batches, left):
                    batch = batch.copy()
                    for category in self._output_categories:
                        batch[category] = batch[category][0:to_copy]
                    output.append(batch)
                return output

        else:
            if self._last_batch_policy == LastBatchPolicy.PARTIAL and (
                                          self._counter > self._size) and self._size > 0:
                # First calculate how much data is required to return exactly self._size entries.
                diff = self._num_gpus * self.batch_size - (self._counter - self._size)
                # Figure out how many GPUs to grab from.
                numGPUs_tograb = int(np.ceil(diff / self.batch_size))
                # Figure out how many results to grab from the last GPU
                # (as a fractional GPU batch may be required to bring us
                # right up to self._size).
                mod_diff = diff % self.batch_size
                data_fromlastGPU = mod_diff if mod_diff else self.batch_size

                # Grab the relevant data.
                # 1) Grab everything from the relevant GPUs.
                # 2) Grab the right data from the last GPU.
                # 3) Append data together correctly and return.
                output = data_batches[0:numGPUs_tograb]
                output[-1] = output[-1].copy()
                for category in self._output_categories:
                    output[-1][category] = output[-1][category][0:data_fromlastGPU]
                return output

        return data_batches
