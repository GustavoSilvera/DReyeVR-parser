import pandas as pd
from typing import Dict, List, Any
from parser import parse_file
from utils import (
    check_for_periph_data,
    convert_to_df,
    split_along_subgroup,
    get_good_idxs,
    fill_gaps,
    compute_YP,
)
from visualizer import plot_versus, plot_histogram2d
import numpy as np
import argparse

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="DReyeVR recording parser")
    argparser.add_argument(
        "-f",
        "--file",
        metavar="P",
        default=None,
        type=str,
        help="path of the (human readable) recording file",
    )
    args = argparser.parse_args()
    filename: str = args.file
    if filename is None:
        print("Need to pass in the recording file")
        exit(1)

    """parse the file"""
    data: Dict[str, np.ndarray or dict] = parse_file(filename)

    """append/generate periph data if available"""
    # check for periph data
    PeriphData = check_for_periph_data(data)
    if PeriphData is not None:
        data["PeriphData"] = PeriphData

    # """convert to pandas df"""
    # # need to split along groups so all data lengths are the same
    # data_groups = split_along_subgroup(data, ["CustomActor"])
    # data_groups_df: List[pd.DataFrame] = [convert_to_df(x) for x in data_groups]

    t: np.ndarray = data["TimestampCarla"]["data"]

    """visualize some interesting data!"""

    eye = data["EyeTracker"]
    all_valid = (
        eye["COMBINEDGazeValid"]
        & eye["LEFTGazeValid"]
        & eye["LEFTEyeOpennessValid"]
        & eye["LEFTPupilPositionValid"]
        & eye["RIGHTGazeValid"]
        & eye["RIGHTEyeOpennessValid"]
        & eye["RIGHTPupilPositionValid"]
    )
    all_valid_idxs = np.where(all_valid == 1)
    print(f"Total validity proportion: {100 * np.sum(all_valid) / len(all_valid):.3f}%")
    frames = t[all_valid_idxs]

    plot_versus(
        data_x=t,
        name_x="Time",
        data_y=all_valid,
        name_y="Confidence (validity)",
        units_y="",
        units_x="s",
        lines=False,
    )

    pupil_mm_L = eye["LEFTPupilDiameter"][all_valid_idxs]
    if (pupil_mm_L < 0).any():  # correct for negatives
        pupil_mm_L = fill_gaps(pupil_mm_L, lambda x: x < 0, mode="mean")
    plot_versus(
        data_x=t,
        name_x="Time",
        data_y=pupil_mm_L,
        name_y="Left pupil diameter",
        units_y="mm",
        units_x="s",
        lines=True,
    )

    pupil_mm_R = eye["RIGHTPupilDiameter"][all_valid_idxs]
    if (pupil_mm_R < 0).any():  # correct for negatives
        pupil_mm_R = fill_gaps(pupil_mm_R, lambda x: x < 0, mode="mean")
    plot_versus(
        data_x=t,
        name_x="Time",
        data_y=pupil_mm_R,
        name_y="Right pupil diameter",
        units_y="mm",
        units_x="s",
        lines=True,
    )

    gaze_dir_C = eye["COMBINEDGazeDir"][all_valid_idxs]
    gaze_yaw_C, gaze_pitch_C = compute_YP(gaze_dir_C)
    plot_histogram2d(
        data_x=gaze_yaw_C,
        data_y=gaze_pitch_C,
        name_x="yaw_C",
        name_y="pitch_C",
        units_x="deg",
        units_y="deg",
        bins=100,
    )
    gaze_dir_L = eye["LEFTGazeDir"][all_valid_idxs]
    gaze_yaw_L, gaze_pitch_L = compute_YP(gaze_dir_L)
    plot_histogram2d(
        data_x=gaze_yaw_L,
        data_y=gaze_pitch_L,
        name_x="yaw_L",
        name_y="pitch_L",
        units_x="deg",
        units_y="deg",
        bins=100,
    )

    gaze_dir_R = eye["RIGHTGazeDir"][all_valid_idxs]
    gaze_yaw_R, gaze_pitch_R = compute_YP(gaze_dir_R)
    plot_histogram2d(
        data_x=gaze_yaw_R,
        data_y=gaze_pitch_R,
        name_x="yaw_R",
        name_y="pitch_R",
        units_x="deg",
        units_y="deg",
        bins=100,
    )
