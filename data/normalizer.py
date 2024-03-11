import os
import re
from typing import Optional, Dict, Tuple, Union, List
import numpy as np
import pickle
from scipy.interpolate import interp1d  


def check_atmos_pattern(s):  
    pattern = r'_\d+$'  
    return bool(re.search(pattern, s))  

class NormStore:
    def __init__(self, normalize_path, 
                 std_file_name: str = "normalize_std2.pkl", 
                 mean_file_name: str = "normalize_mean2.pkl") -> None:
        self.g = 9.80665 # Gravitational acceleration constant
        (self.single_mean, 
         self.single_std, 
         self.atmos_mean_func, 
         self.atmos_std_func) = self._load_norm_files(normalize_path, std_file_name, mean_file_name)
        
        (self.goes_mean, 
         self.goes_std) = (np.array([7.5266e-02, 2.7833e+02, 2.5044e+02, 2.7173e+02]),
                            np.array([ 0.1267, 19.0442, 11.1046, 20.7373]))
        
    def _load_norm_files(
        self, normalize_path: str, 
        std_file_name: str = "normalize_std2.pkl", 
        mean_file_name: str = "normalize_mean2.pkl"
    ) -> Tuple[
        Union[Dict, None],
        Union[Dict, None],
    ]:
        """Get statistics for normalization transforms

        Args:
            normalize_path (str): Path to statistics
            single_vars (tuple[str, ...]): A tuple of used single variables
            atmos_vars (tuple[str, ...]): A tuple of used atmospherical variables

        Returns:
            transforms.Normalize: Normalization for single variables
            transforms.Normalize: Normalization for atmospherical variables
            all_mean: a dict of mean of all used variables
            all_std: a dict of std of all used variables
        """
        mean_path = os.path.join(normalize_path, mean_file_name)
        std_path = os.path.join(normalize_path, std_file_name)
        # mean_path = os.path.join(normalize_path, "normalize_delta_mean.pkl")
        # std_path = os.path.join(normalize_path, "normalize_delta_std.pkl")
        if not os.path.exists(mean_path):
            return None, None, None, None

        with open(mean_path, "rb") as fp:
            normalize_mean = pickle.load(fp)
        with open(std_path, "rb") as fp:
            normalize_std = pickle.load(fp)

        # split single variables and atmos variables
        single_vars, atmos_vars = [], []
        for k in normalize_mean.keys():
            if check_atmos_pattern(k):
                atmos_vars.append(k)
            else:
                single_vars.append(k)
        single_mean = {var: normalize_mean[var] for var in single_vars}
        single_std = {var: normalize_std[var] for var in single_vars}

        atmos_variables = {v.rsplit("_", 1)[0] for v in atmos_vars}
        atmos_levels: dict[str, list[int]] = {k: [] for k in atmos_variables}
        for var in atmos_vars:
            atmos_levels[var.rsplit("_", 1)[0]].append(int(var.rsplit("_", 1)[1]))
        for var in atmos_levels:
            atmos_levels[var] = np.array(sorted(atmos_levels[var]))

        atmos_mean: dict[str, list] = {k: [] for k in atmos_variables}
        atmos_std: dict[str, list] = {k: [] for k in atmos_variables}
        atmos_mean_func, atmos_std_func = {}, {}
        for k in atmos_variables:
            for level in atmos_levels[k]:
                atmos_mean[k].append(normalize_mean[f"{k}_{level}"])
                atmos_std[k].append(normalize_std[f"{k}_{level}"])
                # all_mean[f"{k}_{level}"] = torch.FloatTensor(normalize_mean[f"{k}_{level}"])
                # all_std[f"{k}_{level}"] = torch.FloatTensor(normalize_std[f"{k}_{level}"])

            atmos_mean[k] = np.array(atmos_mean[k])
            atmos_std[k] = np.array(atmos_std[k])

            atmos_mean_func[k] = interp1d(atmos_levels[k], atmos_mean[k])
            atmos_std_func[k] = interp1d(atmos_levels[k], atmos_std[k])

            if k == 'geopotential' and 'geopotential_height' not in atmos_variables:
                # add geopotential height: geopotential_height = geopotential/g
                atmos_mean_func['geopotential_height'] = interp1d(atmos_levels[k], atmos_mean[k]/self.g)
                atmos_std_func['geopotential_height'] = interp1d(atmos_levels[k], atmos_std[k]/self.g)

        return single_mean, single_std, atmos_mean_func, atmos_std_func

    def get_atmos_norm(self, atmos, levels):
        levels = np.array(levels)
        means = self.atmos_mean_func[atmos](levels)
        stds = self.atmos_std_func[atmos](levels)
        return {"mean": means, "std": stds}

    def get_single_norm(self, single):
        return {"mean": self.single_mean[single], "std": self.single_std[single]}

    def get_goes_norm(self):
        return {"mean": self.goes_mean, "std": self.goes_std}
        
if __name__ == "__main__":
    normalize_path = './'
    ns = NormStore(normalize_path)
    print(ns.get_atmos_norm("temperature", [100, 105, 200]))
    print(ns.get_single_norm('2m_temperature'))
    print("...")