import os
from typing import Dict, List, Any, Optional, Tuple
import time
import pandas as pd


def flatten_dict(
    data: Dict[str, Any], sep: Optional[str] = "_"
) -> Dict[str, List[Any]]:
    flat = {}
    for k in data.keys():
        if isinstance(data[k], dict):
            k_flat = flatten_dict(data[k])
            for kd in k_flat.keys():
                key: str = f"{k}{sep}{kd}"
                flat[key] = k_flat[kd]
        else:
            flat[k] = data[k]
    # check no lingering dictionaies
    for k in flat.keys():
        assert not isinstance(flat[k], dict)
    return flat


def split_along_subgroup(
    data: Dict[str, Any], subgroups: List[str]
) -> Tuple[Dict[str, Any]]:
    # splits the one large dict along subgroups such as DReyeVR core and custom-actor data
    # TODO: make generalizable for arbitrary subgroups!
    ret = []
    for sg in subgroups:
        assert sg in data.keys()
        ret.append({sg: data.pop(sg)})  # include the sub-data as its own dict
    ret.append(data)  # include all other data as its own "default" group
    return tuple(ret)


def process_UE4_string_to_value(value: str) -> Any:
    ret = value
    if (
        "X=" in value  # x
        or "Y=" in value  # y (or yaw)
        or "Z=" in value  # z
        or "P=" in value  # pitch
        or "R=" in value  # roll
    ):
        # coming from an FVector or FRotator
        raw_data = value.replace("=", " ").split(" ")
        ret = [
            process_UE4_string_to_value(elem) for elem in raw_data[1::2]
        ]  # every *other* odd
    else:
        try:
            ret = eval(value)
        except Exception:
            ret = value
    return ret
