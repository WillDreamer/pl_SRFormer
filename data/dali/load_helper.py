import os
import torch
# import clip  
import numpy as np
import torch.nn.functional as F  
import pickle
from typing import Dict
from data.normalizer import NormStore
from data.variable_mapping.variable_mapping import VariableMappingInfo
import time

def extract_mean_std(name : str = 'hrrr',
                     sfc_vars : list = ['var'],
                     prs_vars : list = ['var'],
                     prs_levels : list = [0],
                     norm_path : str = './',
                     data_type = torch.float16,
                     std_file_name: str = "normalize_std2.pkl", 
                     mean_file_name: str = "normalize_mean2.pkl"
                     ):
    '''
    Extract mean and std from the norm dict from data dict, the 'std_dict' should only contain one single dataset.
    
    Args:
        std_dict: The norm dict from data dict, eg. data_dict['norm']['era5']
        single_vars: The single variables list
        atmos_vars: The atmospheric variables list
        atmos_levels: The atmospheric levels list, NOTE: this list contains float numbers, which need to be converted to int.
    
    Returns:
        NOTE: For atmos variables, key is f"{atmos_var}_{int(level)}"
        all_mean: A dict contains all mean values of variables
        all_std: A dict contains all std values of variables
    '''
    vm = VariableMappingInfo()

    all_mean = {}
    all_std = {}
    norm_store = NormStore(norm_path, std_file_name, mean_file_name)
    
    # 0.goes means and stds
    if name == 'goes':
        goes_mean = torch.from_numpy(norm_store.get_goes_norm()['mean'])
        goes_std = torch.from_numpy(norm_store.get_goes_norm()['std'])
        all_mean['all_data'] = goes_mean
        all_std['all_data'] = goes_std
        return all_mean, all_std
    
    elif name in ['era5', 'hrrr', 'hrrr_delta']:
        if name == 'hrrr_delta':
            name = 'hrrr'
        # 1.Era5 & Hrrr means and stds
        pressure_levels = [vm.convert_variable(level, 'canonical_short_name', 'atmos_levels', f'{name.upper()}_long') for level in prs_levels]
        if name == 'hrrr':
            pressure_levels = [int(p/100.) for p in pressure_levels]
        elif name == 'era5':
            pressure_levels = [int(p) for p in pressure_levels]
            
        for var in sfc_vars:
            var_in_norm = vm.convert_variable(var, 'canonical_short_name', 'surf_vars', 'ERA5_long')
            mean_std = norm_store.get_single_norm(var_in_norm)
            if type(mean_std['mean']) == np.ndarray:
                all_mean[var] = torch.from_numpy(mean_std['mean'])
                all_std[var] = torch.from_numpy(mean_std['std'])
            elif type(mean_std['mean']) == np.float64:
                all_mean[var] = torch.tensor(mean_std['mean'])
                all_std[var] = torch.tensor(mean_std['std'])

        # 1.1 Surface data means and stds
        all_mean['surface_data'] = torch.stack([all_mean[var] for var in sfc_vars])
        all_std['surface_data'] = torch.stack([all_std[var] for var in sfc_vars])

        for var in prs_vars:
            # NOTE:This is very DIRTY!!! Cause canonical_short_name 'hgtn' cann't be convert into a ERA5 long name.
            if var == 'hgtn':
                var_in_norm = 'geopotential_height'
            else:
                var_in_norm = vm.convert_variable(var, 'canonical_short_name', 'atmos_vars', 'ERA5_long')

            mean_std = norm_store.get_atmos_norm(var_in_norm, pressure_levels)
            all_mean[var] = torch.from_numpy(mean_std['mean'])
            all_std[var] = torch.from_numpy(mean_std['std'])
            for idx, level in enumerate(prs_levels):
                all_mean[f"{var}_{int(level)}"] = torch.from_numpy(mean_std['mean'][idx:idx+1])
                all_std[f"{var}_{int(level)}"] = torch.from_numpy(mean_std['std'][idx:idx+1])
        
        # 1.2 Atmospheric data means and stds
        all_mean['atmos_data'] = torch.cat([all_mean[f"{var}_{int(level)}"] for var in prs_vars for level in prs_levels])
        all_std['atmos_data'] = torch.cat([all_std[f"{var}_{int(level)}"] for var in prs_vars for level in prs_levels])
            
        all_mean['all_data'] = torch.cat([all_mean['surface_data'], all_mean['atmos_data']])
        all_std['all_data'] = torch.cat([all_std['surface_data'], all_std['atmos_data']])

        return all_mean, all_std


def get_inputs(batch,
               era5_norm=None,
               hrrr_norm=None,
               goes_norm=None,
               need_mrms=False,
               need_station=False,
               data_type=torch.float32,
               picked_idx=None,
               ):
    device = batch['input_hrrr']['surface'].device if 'input_hrrr' in batch.keys() else batch['input_era5']['surface'].device
    top_left = batch['meta_info']['top_left']
    batch_size = top_left[0].shape[0]
    top_left = [torch.LongTensor([top_left[0][i]]).to(device) for i in range(batch_size)], \
                [torch.LongTensor([top_left[1][i]]).to(device) for i in range(batch_size)]
    dt = batch['meta_info']['timestamp'][0]    
    print(f"timestamp: {dt.year}-{dt.month}-{dt.day}-{dt.hour}")
    # top_left: {top_left[0].item()}-{top_left[1].item()}")
    timestamp = [torch.LongTensor([dt.year]).to(device), 
                 torch.LongTensor([dt.month]).to(device), 
                 torch.LongTensor([dt.day]).to(device), 
                 torch.LongTensor([dt.hour]).to(device)]
    # timestamp = None
    
    inputs, outputs = [], []
    
    # Era5 if required, e.g. (batch_size, 2, 69, 72, 72)
    if era5_norm is not None:
        era5_vars_mean, era5_vars_std = era5_norm
        
        era5_vars_mean = era5_vars_mean[None, None, :, None, None].to(device)
        era5_vars_std = era5_vars_std[None, None, :, None, None].to(device)
        
        input_era5 = (torch.cat([batch['input_era5']['surface'], batch['input_era5']['pressure']], dim=2) - era5_vars_mean) / era5_vars_std
        output_era5 = (torch.cat([batch['output_era5']['surface'], batch['output_era5']['pressure']], dim=2) - era5_vars_mean) / era5_vars_std
        
        # input_era5 = input_era5[:,:,:,16:16+64,16:16+64]
        # output_era5 = output_era5[:,:,:,16:16+64,16:16+64]

        input_era5 = input_era5.to(data_type)
        output_era5 = output_era5.to(data_type)
        
        # any nan in era5 data will be raise error
        if torch.isnan(input_era5).any() or torch.isnan(output_era5).any():
            raise ValueError('Era5 data contains nan value!')
        
        inputs.append(input_era5)
        outputs.append(output_era5)
    else:
        inputs.append(None)
        outputs.append(None)
    
    # Hrrr if required, e.g. (batch_size, 2, 69, 640, 640)
    if hrrr_norm is not None:
        hrrr_vars_mean, hrrr_vars_std = hrrr_norm
        
        hrrr_vars_mean = hrrr_vars_mean[None, None, :, None, None].to(device)
        hrrr_vars_std = hrrr_vars_std[None, None, :, None, None].to(device)

        input_hrrr = (torch.cat([batch['input_hrrr']['surface'], batch['input_hrrr']['pressure']], dim=2) - hrrr_vars_mean) / hrrr_vars_std
        output_hrrr = (torch.cat([batch['output_hrrr']['surface'], batch['output_hrrr']['pressure']], dim=2) - hrrr_vars_mean) / hrrr_vars_std

        input_hrrr = input_hrrr.to(data_type)
        output_hrrr = output_hrrr.to(data_type)

        # any nan in hrrr data will be raise error
        if torch.isnan(input_hrrr).any() or torch.isnan(output_hrrr).any():
            raise ValueError('Hrrr data contains nan value!')
        
        inputs.append(input_hrrr)
        outputs.append(output_hrrr)
    else:
        inputs.append(None)
        outputs.append(None)
        
    # Goes if required, e.g. (batch_size, 12, 4, 640, 640)
    if goes_norm is not None and 'input_goes' in batch.keys() and 'output_goes' in batch.keys():
        goes_vars_mean, goes_vars_std = goes_norm
        
        goes_vars_mean = goes_vars_mean[None, None, :, None, None].to(device)
        goes_vars_std = goes_vars_std[None, None, :, None, None].to(device)

        input_goes = (batch['input_goes'] - goes_vars_mean) / goes_vars_std
        output_goes = (batch['output_goes'] - goes_vars_mean) / goes_vars_std
        
        input_goes = input_goes.to(data_type)
        output_goes = output_goes.to(data_type)
        
        # any nan in goes data will be raise error
        if torch.isnan(input_goes).any() or torch.isnan(output_goes).any():
            raise ValueError('Goes data contains nan value!')
        
        inputs.append(input_goes)
        outputs.append(output_goes)
    else:
        inputs.append(None)
        outputs.append(None)
        
    # Mrms if required, e.g.  (batch_size, 12, 1920, 1920)
    if need_mrms and 'input_mrms' in batch.keys() and 'output_mrms' in batch.keys():
        input_mrms = batch['input_mrms'].to(data_type)
        output_mrms = batch['output_mrms'].to(data_type)
        
        # any nan in mrms data will be raise error
        if torch.isnan(input_mrms).any() or torch.isnan(output_mrms).any():
            raise ValueError('Mrms data contains nan value!')
        
        inputs.append(input_mrms)
        outputs.append(output_mrms)
    else:
        inputs.append(None)
        outputs.append(None)
        
    # Station if required, e.g. (batch_size, 120, 5, 640, 640)
    if need_station and 'input_station' in batch.keys() and 'output_station' in batch.keys():
        input_station = batch['input_station'].to(data_type)
        output_station = batch['output_station'].to(data_type)
        
        # any nan in station data will be raise error
        if torch.isnan(input_station).any() or torch.isnan(output_station).any():
            raise ValueError('Station data contains nan value!')
        
        inputs.append(input_station)
        outputs.append(output_station)
    else:
        inputs.append(None)
        outputs.append(None)

    if picked_idx is not None and inputs[1] is not None:
        if inputs[0] is not None:
            inputs[0] = inputs[0][:, :, picked_idx, :, :]
            outputs[0] = outputs[0][:, :, picked_idx, :, :]
        if inputs[1] is not None:
            inputs[1] = inputs[1][:, :, picked_idx, :, :]
            outputs[1] = outputs[1][:, :, picked_idx, :, :]
    return inputs, outputs, top_left, timestamp