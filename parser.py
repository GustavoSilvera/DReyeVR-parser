import os
from typing import Dict, Iterable, List, Any, Optional, Tuple
import time
import pandas as pd
from utils import process_UE4_string_to_value, split_along_subgroup, flatten_dict
import numpy as np

# used as the dictionary key when the data has no explicit title (ie. included as raw array)
_no_title_key: str = "data"


def parse_row(
    data: Dict[str, Any],
    data_line: str,
    title: Optional[str] = "",
    t: Optional[int] = 0,
) -> None:
    # cleanup data line
    data_line: List[str] = data_line.replace("}", "{").replace("{", ",").split(",")

    if title == "":  # find title with the data line
        title: str = data_line[0].split(":")[0]
        # remove title for first elem
        data_line[0] = data_line[0].replace(f"{title}:", "")
    working_map: Dict[str, Any] = {} if title not in data else data[title]
    data[title] = working_map  # ensure this working set contributes to the larger set

    if t != 0:
        if "t" not in working_map:
            working_map["t"] = []
        working_map["t"].append(t)
    subtitle: str = ""  # subtitle for elements within the dictionary
    for element in data_line:

        # case when empty string occurs (from "{" or "}" splitting)
        if element == "":
            subtitle = ""  # reset subtitle name
            continue

        # case when only element (after title) is present (eg. TimestampCarla)
        if ":" not in element:
            if _no_title_key not in working_map:
                working_map[_no_title_key] = []
            working_map[_no_title_key].append(process_UE4_string_to_value(element))
            continue

        # common case
        key_value: List[str] = element.split(":")
        key_value: List[str] = [elem for elem in key_value if len(elem) > 0]
        if len(key_value) == 1:  # contains only sublabel, like "COMBINED"
            subtitle = key_value[0]
        elif len(key_value) == 2:  # typical key:value pair
            key, value = key_value
            key = f"{subtitle}{key}"
            value = process_UE4_string_to_value(value)  # evaluate what we think this is

            # add to the working map
            if key not in working_map:
                working_map[key] = []
            working_map[key].append(value)


def validate(data: Dict[str, Any], L: Optional[int] = None) -> None:
    # verify the data structure is reasonable
    if L is None:
        L: int = len(data["TimestampCarla"][_no_title_key])
    for k in data.keys():
        if k == "CustomActor":
            # TODO
            continue
        if isinstance(data[k], dict):
            validate(data[k], L)
        elif isinstance(data[k], list):
            assert len(data[k]) == L or len(data[k]) == L - 1
        else:
            raise NotImplementedError


def parse_file(
    path: str, return_df: Optional[bool] = True
) -> Dict[str, Any] or List[pd.DataFrame]:
    assert os.path.exists(path)
    print(f"Reading DReyeVR recording file: {path}")

    data: Dict[str, List[Any]] = {}

    DReyeVR_core: str = "  [DReyeVR]"
    DReyeVR_CA: str = "  [DReyeVR_CA]"
    with open(path) as f:
        start_t: float = time.time()
        for i, line in enumerate(f.readlines()):

            # checking the line(s) for core DReyeVR data
            if line[: len(DReyeVR_core)] == DReyeVR_core:
                data_line: str = line.strip(DReyeVR_core).strip("\n")
                parse_row(data, data_line)
                validate(data)

            # checking the line(s) for DReyeVR custom actor data
            elif line[: len(DReyeVR_CA)] == DReyeVR_CA:
                data_line: str = line.strip(DReyeVR_CA).strip("\n")
                t = data["TimestampCarla"][_no_title_key][-1]  # get carla time
                parse_row(data, data_line, title="CustomActor", t=t)

            # print status
            if i % 500 == 0:
                t: float = time.time() - start_t
                print(f"Lines read: {i} @ {t:.3f}s", end="\r", flush=True)

    n: int = len(data["TimestampCarla"][_no_title_key])
    print(f"successfully read {n} frames in {t:.3f}s")
    if return_df:
        # need to split along groups so all data lengths are the same
        data_groups = split_along_subgroup(data, ["CustomActor"])
        data_groups_df: List[pd.DataFrame] = [data_to_df(x) for x in data_groups]
        return data_groups_df
    return data


def data_to_df(data: Dict[str, Any]) -> pd.DataFrame:
    data = flatten_dict(data)
    lens = [len(x) for x in data.values()]
    assert min(lens) == max(lens)  # all lengths are equal!
    df = pd.DataFrame.from_dict(data)
    return df
