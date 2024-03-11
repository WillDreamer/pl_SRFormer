import os
import json
import inspect
import pandas as pd
from typing import Dict, List, Type, TypeVar, Any, Optional
VARIABLE_DICTS_ROOT = 'data/variable_mapping'

class VariableMappingInfo:
    """VariableMappingInfo

    A class designed to hold data from the VariableMapping.csv in a structured way,
    and access for a single dataset.

    Can be used for more flexible conversions between all pairs iof variable names if required.
    
    Three methods `list_var_names` and `convert_variable`, `convert_variable_list` are the fundamental methods for internal use. They may not convient for user. 
    
    Three methods `get_canonical_short_name_list`, `get_canonical_short_name_from_dataset_short_name` and `get_dataset_short_name_from_canonical_short_name` are the main methods for user.
    
    These three methods have its own short name function:
    `get_namelist`, `get_canonicalname`, `get_datasetname`. 
    
    """

    mapping_df: pd.DataFrame
    variable_columns: list[str]

    # @classmethod
    def __init__(self, filepath: Optional[str] = None):
        """Load the csv file that contains the mapping information"""
        if filepath:
            assert os.path.exists(filepath), f"The path '{filepath}' does not exist."
            mapping_df = pd.read_csv(filepath)
        else:
            mapping_file_path = os.path.join(VARIABLE_DICTS_ROOT, "variable_mapping.csv")
            assert os.path.exists(
                mapping_file_path
            ), f"The path '{mapping_file_path}' does not exist."
            mapping_df = pd.read_csv(mapping_file_path)
        self.mapping_df = mapping_df
        
        # variable_columns = list(set(mapping_df.columns)  

        # return cls(mapping_df, variable_columns=variable_columns)

    def list_var_names(
        self, from_where: str, variable_type: str,
    ) -> List[str]:
        """List all the variable names in the `type` format from the `from_where`  using the mapping file
        Args:
            from_where: 'canonical_long_name' or 'canonical_short_name' or DatasetName(e.g. 'HRRR_short', 'ERA5_long')
            type: 'surf_vars' or 'atmos_vars' or 'atmos_levels'
        """
        var_names = self.mapping_df[self.mapping_df["var_type"] == variable_type][from_where]
        var_names = var_names.dropna()
        if variable_type == 'atmos_levels':
            output_type = float
        else:
            output_type = str
        return [output_type(x) for x in var_names]

    def convert_variable(
        self, variable: str, from_where: str, from_type: str, to_where: str, 
    ) -> str:
        """Get the variable name in the `to_where` format from the `from_where` format using the mapping file
        Args:   
            variable: The variable name you want to transfer from e.g. '2d' if you want to transfer from 'canonical_short_name'
            from_where: 'canonical_long_name' or 'canonical_short_name' or DatasetName_short(e.g. 'HRRR', 'ERA5')
            from_type: 'surf_vars' or 'atmos_vars' or 'atmos_levels'    
            to_where: 'canonical_long_name' or 'canonical_short_name' or DatasetName(e.g. 'HRRR', 'ERA5')
            
    
        """
        
        this_type_df = self.mapping_df[
           (self.mapping_df["var_type"] == from_type)
        ]
        if from_type == 'atmos_levels':
            variable = float(variable)
            this_value_df = this_type_df[this_type_df[from_where].astype(float) == variable]
        else:
            this_value_df = this_type_df[this_type_df[from_where] == variable]
        var_name = this_value_df[to_where]
        res = var_name.unique()
 
        assert len(res) == 1
        if from_type == 'atmos_levels':
            res = float(res[0])
        else:
            res = res[0]
        return res

    def convert_variable_list(
        self, variable_list: List[str], from_where: str, from_type: str, to_where: str, 
    ) -> List[str]:
        """Get the variable name in the `to_where` format from the `from_where` format using the mapping file
        Args:
            from_where: 'canonical_long_name' or 'canonical_short_name' or DatasetName(e.g. 'HRRR', 'ERA5')
            to_where: 'canonical_long_name' or 'canonical_short_name' or DatasetName(e.g. 'HRRR', 'ERA5')
            from_name: The variable name you want to transfer from e.g. '2d' if you want to transfer from 'canonical_short_name'
        """
        res = []
        for variable in variable_list:
            res.append(self.convert_variable(variable, from_where, from_type, to_where))
        return res
  
    def get_canonical_short_name_list(self,
                                      dataset_name: str):
        """Get the canonical_short_name list of a dataset
        Args:
            dataset_name: DatasetName(e.g. 'HRRR', 'ERA5')
        """
        surf_vars = self.convert_variable_list(
            variable_list = self.list_var_names(from_where=f'{dataset_name}_short', variable_type='surf_vars' ),
            from_where=f'{dataset_name}_short', from_type='surf_vars', to_where='canonical_short_name')
        atmos_vars = self.convert_variable_list(
            variable_list=self.list_var_names(from_where=f'{dataset_name}_short', variable_type='atmos_vars' ),
            from_where=f'{dataset_name}_short', 
            from_type='atmos_vars', to_where='canonical_short_name')
        atmos_levels = self.convert_variable_list(
            variable_list=self.list_var_names(from_where=f'{dataset_name}_short', variable_type='atmos_levels' ),
            from_where=f'{dataset_name}_short', 
            from_type='atmos_levels', to_where='canonical_short_name')
 
        return [surf_vars, atmos_vars,atmos_levels]
    
    def get_canonical_short_name_from_dataset_short_name(self,
        dataset_name: str,
        variable_name: str,
        variable_type: str):
        """Get the canonical_short_name of a variable in a dataset
        Args:
            dataset_name: DatasetName(e.g. 'HRRR', 'ERA5')
            variable_name: variable name in the dataset
            variable_type: 'surf_vars' or 'atmos_vars' or 'atmos_levels'
        """
        return self.convert_variable(                             
            variable=variable_name, 
            from_where=f'{dataset_name}_short', from_type=variable_type, to_where='canonical_short_name')
    
    def get_dataset_short_name_from_canonical_short_name(self,  
        dataset_name: str,
        variable_name: str,        
        variable_type: str):
        """Get the dataset_short_name of a variable in a dataset
        Args:
            canonical_short_name: canonical_short_name of a variable
            dataset_name: DatasetName(e.g. 'HRRR', 'ERA5')
            variable_type: 'surf_vars' or 'atmos_vars' or 'atmos_levels'
        """
        return self.convert_variable(variable_name, from_where='canonical_short_name', from_type=variable_type, to_where=f'{dataset_name}_short')
    
    def get_namelist(self, dataset_name: str):
        """Get the namelist of a dataset
        Short name version for self.get_canonical_short_name_list
        Args:
            dataset_name: DatasetName(e.g. 'HRRR', 'ERA5')
        """
        return self.get_canonical_short_name_list(dataset_name = dataset_name) 
    
    def get_canonicalname(
        self,
        dataset_name: str,
        variable_name: str,
        variable_type: str):
        """Get the canonical_short_name of a variable in a dataset
        Short name version for self.get_canonical_short_name_from_dataset_short_name
        Args:
            dataset_name: DatasetName(e.g. 'HRRR', 'ERA5')
            variable_name: variable name in the dataset
            variable_type: 'surf_vars' or 'atmos_vars' or 'atmos_levels'
        """
        return self.get_canonical_short_name_from_dataset_short_name(
            dataset_name=dataset_name,
            variable_name=variable_name,
            variable_type=variable_type)
        
    def get_datasetname(
        self,
        dataset_name: str,
        variable_name: str,
        variable_type: str):
        """
        Short name version for self.get_dataset_short_name_from_canonical_short_name
        Args:
            canonical_short_name: canonical_short_name of a variable
            dataset_name: DatasetName(e.g. 'HRRR', 'ERA5')
            variable_type: 'surf_vars' or 'atmos_vars' or 'atmos_levels'
        """
        return self.get_dataset_short_name_from_canonical_short_name(            
            dataset_name=dataset_name,
            variable_name=variable_name,
            variable_type=variable_type)
        
 
    
if __name__ == '__main__':


    vm = VariableMappingInfo('./variable_mapping.csv')
    print(vm.get_namelist('HRRR'))
    print(vm.get_namelist('ERA5'))
    print(vm.get_datasetname('HRRR', 'sp', 'surf_vars'))
    print(vm.get_canonicalname('HRRR', 'PRES_P0_L1_GLC0', 'surf_vars'))
    print(vm.get_datasetname('HRRR', vm.get_canonicalname('HRRR', '5000', 'atmos_levels'), 'atmos_levels'))
    print(vm.get_canonicalname('HRRR', vm.get_datasetname('HRRR', '5000', 'atmos_levels'), 'atmos_levels'))
    print(vm.get_canonicalname('HRRR', 5000, 'atmos_levels'))
    print(vm.get_canonicalname('HRRR', '5000', 'atmos_levels'))
    print(vm.get_canonicalname('HRRR', '5000.0', 'atmos_levels'))