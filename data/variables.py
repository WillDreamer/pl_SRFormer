# mypy: ignore-errors

single_vars = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
]
test_vars = [
    "2m_temperature",
    "temperature",
]

import os
import json
import inspect
from dataclasses import dataclass, asdict
from typing import Dict, List, Type, TypeVar, Any

from paths import CLIMAI_GLOBAL_ROOT

VariableMapType = TypeVar("VariableMapType", bound="VariableMapping")

VARIABLE_DICTS_ROOT = os.path.join(CLIMAI_GLOBAL_ROOT, "data", "variable_mapping")

DATASET_NAME_MAP = {
    "cmcc": "cmcc-cm2-vhr4",
    "ecwmf": "ecwmf-ifs-hr",
    "era5": "era5",
    "era5pangu": "era5-pangu",
}

# TODO(Megan): currently running on test variables does not work for era5, fix.
TEST_VARIABLES = ["ta", "tas"]


@dataclass(frozen=True, order=False)
class VariableMapping:
    name: str
    surface_variables: Dict[str, str]
    surface_variables_conversion: Dict[str, str]
    atmospheric_variables: Dict[str, str]
    atmospheric_variables_conversion: Dict[str, str]
    pressure_levels: List[int]

    def __post_init__(self):
        # TODO(general): put some checks here. eg. warn if there is going to be a variable name clash
        pass

    @classmethod
    def _invert_var_dict(cls: Type[VariableMapType], data: Dict[str, str]) -> Dict[str, str]:
        return {v: k for k, v in data.items()}

    @classmethod
    def _filter_dict(
        cls: Type[VariableMapType], data: Dict[str, str], filter_vars: List[str]
    ) -> Dict[str, str]:
        return {k: v for k, v in data.items() if k in filter_vars}

    @property
    def variables(self) -> Dict[str, str]:
        # get combined surface and atmospheric variables dict with unique shortnames as keys
        inverted_surface_vars = self._invert_var_dict(self.surface_variables)
        inverted_atmospheric_vars = self._invert_var_dict(self.atmospheric_variables)
        return dict(inverted_surface_vars, **inverted_atmospheric_vars)

    @property
    def longname_variables(self) -> Dict[str, str]:
        # combined surface and atmospheric variables dict as read in from json file
        return dict(**self.surface_variables, **self.atmospheric_variables)

    @classmethod
    def filter_loaded_data(
        cls: Type[VariableMapType], data: Dict[str, Any], filter_vars: List[str]
    ) -> Dict[str, Any]:
        # selects out the relevant bits of the data dict for this variable set
        return {
            "surface_variables": cls._invert_var_dict(
                cls._filter_dict(cls._invert_var_dict(data["surface_variables"]), filter_vars)
            ),
            "atmospheric_variables": cls._invert_var_dict(
                cls._filter_dict(cls._invert_var_dict(data["atmospheric_variables"]), filter_vars)
            ),
            "surface_variables_conversion": cls._filter_dict(
                data["surface_variables_conversion"], filter_vars
            ),
            "atmospheric_variables_conversion": cls._filter_dict(
                data["atmospheric_variables_conversion"], filter_vars
            ),
            "pressure_levels": data["pressure_levels"],
        }

    @classmethod
    def from_name(
        cls: Type[VariableMapType], dataset_name: str, restricted_test_variables: bool = False
    ) -> VariableMapType:
        # returns an instance of the class from the name
        dataset_full_name = DATASET_NAME_MAP[dataset_name.lower()].replace("-", "_")
        data = {"name": dataset_full_name}
        with open(os.path.join(VARIABLE_DICTS_ROOT, f"{dataset_full_name}.json"), "r") as f:
            loaded_data = json.load(f)
        if restricted_test_variables:
            data.update(cls.filter_loaded_data(loaded_data, TEST_VARIABLES))
        else:
            data.update(loaded_data)

        return cls(
            **{
                key: value
                for key, value in data.items()
                if key in inspect.signature(cls).parameters
            }
        )

    @property
    def mapping(self) -> Dict[str, str]:
        # return the full map from short name to canonical shortname
        return dict(self.surface_variables_conversion, **self.atmospheric_variables_conversion)

    @property
    def surface_shortnames(self) -> List[str]:
        return list(self.surface_variables.values())

    @property
    def atmospheric_shortnames(self) -> List[str]:
        return list(self.atmospheric_variables.values())

    @property
    def shortnames(self) -> List[str]:
        return list(self.variables.keys())

    @property
    def longnames(self) -> List[str]:
        return list(self.variables.values())

    def get_longname(self, short_var: str) -> str:
        # return the long name corresponding to a shortname code
        # with an extension to distinguish surface or atmosphere
        if short_var in self.surface_shortnames:
            long_var = f"surface_{self.variables[short_var]}"
        elif short_var in self.atmospheric_shortnames:
            long_var = f"atmosphere_{self.variables[short_var]}"
        return long_var

    def get_shortname(self, long_var: str) -> str:
        # return the short name code corresponding to a long name
        # this does not assume that eg. era5 is canonical -- for now each
        # dataset file individually makes the mapping between shortnames and canonical names

        if long_var in self.surface_variables:
            short_var = self.surface_variables[long_var]
        elif long_var in self.atmospheric_variables:
            short_var = self.atmospheric_variables[long_var]
        return short_var

    def to_json(self) -> None:
        # write out data from an instance of the class to file
        with open(os.path.join(VARIABLE_DICTS_ROOT, f"{self.name}.json"), "w") as f:
            json.dump(asdict(self), f, indent=4)

    def convert(self, var: str) -> str:
        # convert variable into the standardised form
        # must be short variable name as long names are not unique
        if var in self.longnames:
            raise ValueError(f"{var} is not a variable shortname and cannot be uniquely specified.")
        return self.mapping[var]
