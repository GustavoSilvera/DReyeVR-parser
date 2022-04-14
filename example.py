import pandas as pd
from typing import Dict, List, Any
from parser import parse_file
from utils import check_for_periph_data, convert_to_df, split_along_subgroup
import numpy as np

filename: str = "/Users/gustavo/carla/carla.mac/PythonAPI/examples/recorder3.txt"

if __name__ == "__main__":
    """parse the file"""
    data: Dict[str, np.ndarray or dict] = parse_file(filename)

    """append/generate periph data if available"""
    # check for periph data
    PeriphData = check_for_periph_data(data)
    if PeriphData is not None:
        data["PeriphData"] = PeriphData

    """convert to pandas df"""
    # need to split along groups so all data lengths are the same
    data_groups = split_along_subgroup(data, ["CustomActor"])
    data_groups_df: List[pd.DataFrame] = [convert_to_df(x) for x in data_groups]
