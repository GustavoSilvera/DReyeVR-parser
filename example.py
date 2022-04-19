from matplotlib.pyplot import axes
import pandas as pd
from typing import Dict, List, Any
from parser import parse_file
from utils import (
    check_for_periph_data,
    convert_to_df,
    split_along_subgroup,
    get_good_idxs,
    smooth_arr,
    fill_gaps,
    compute_YP,
)
from visualizer import (
    plot_versus,
    plot_histogram2d,
    plot_vector_vs_time,
    plot_3Dt,
)
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

    t: np.ndarray = data["TimestampCarla"]["data"] / 1000  # to seconds
    # now t is in seconds

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
    eye_valid_t = t[all_valid_idxs]

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
        data_x=eye_valid_t,
        name_x="Time",
        data_y=smooth_arr(pupil_mm_L, 100),
        name_y="Left pupil diameter",
        units_y="mm",
        units_x="s",
        lines=True,
    )

    pupil_mm_R = eye["RIGHTPupilDiameter"][all_valid_idxs]
    if (pupil_mm_R < 0).any():  # correct for negatives
        pupil_mm_R = fill_gaps(pupil_mm_R, lambda x: x < 0, mode="mean")
    plot_versus(
        data_x=eye_valid_t,
        name_x="Time",
        data_y=smooth_arr(pupil_mm_R),
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

    """plot 3D position over time"""
    pos3D = data["EgoVariables"]["VehicleLoc"]
    plot_vector_vs_time(pos3D, t, "EgoPos XYZ")
    # found that the first ~50 is kinda garbage, just omit them
    omit_front: int = 50
    plot_vector_vs_time(pos3D[omit_front:], t[omit_front:], "EgoPos XYZ (50:)")

    plot_3Dt(
        xyz=pos3D[omit_front:],
        t=t[omit_front:],
        title="Vehicle position over time",
        interactive=False,  # set to True to move it around
    )

    """plot pupil position"""
    pupil_pos_L = eye["LEFTPupilPosition"][all_valid_idxs]
    plot_vector_vs_time(pupil_pos_L, eye_valid_t, "Left pupil position")
    plot_histogram2d(
        data_x=pupil_pos_L[:, 0],
        data_y=pupil_pos_L[:, 1],
        name_x="LPupilX",
        name_y="LPupilY",
        bins=100,
    )

    pupil_pos_R = eye["RIGHTPupilPosition"][all_valid_idxs]
    plot_vector_vs_time(pupil_pos_R, eye_valid_t, "Right pupil position")
    plot_histogram2d(
        data_x=pupil_pos_R[:, 0],
        data_y=pupil_pos_R[:, 1],
        name_x="RPupilX",
        name_y="RPupilY",
        bins=100,
    )

    """plot eye vars"""
    plot_versus(
        data_x=eye_valid_t,
        name_x="Time",
        data_y=eye["LEFTEyeOpenness"],
        name_y="Left eye openness",
        units_y="mm",
        units_x="s",
        lines=True,
    )

    plot_versus(
        data_x=eye_valid_t,
        name_x="Time",
        data_y=eye["RIGHTEyeOpenness"],
        name_y="Right eye openness",
        units_y="mm",
        units_x="s",
        lines=True,
    )

    """compute intrinsic factors"""
    delta_ts = np.diff(t)  # t is in seconds
    n: int = len(delta_ts)
    assert delta_ts.min() > 0  # should always be monotonically increasing!
    ego_displacement = np.diff(data["EgoVariables"]["VehicleLoc"], axis=0)
    ego_velocity = (ego_displacement.T / delta_ts).T
    assert ego_velocity.shape == (n, 3)
    cmps2mph = 0.0223694  # cm/s to mph
    speed = np.linalg.norm(ego_velocity, axis=1)  # velocity (3D) to speed (1D)
    plot_versus(
        data_x=t[1:],
        name_x="Time",
        data_y=cmps2mph * speed,
        name_y="Ego Speed",
        units_y="mph",
        units_x="s",
        lines=True,
    )

    ego_accel = (np.diff(ego_velocity, axis=0).T / delta_ts[1:]).T
    assert ego_accel.shape == (n - 1, 3)
    plot_versus(
        data_x=t[2:],
        name_x="Time",
        data_y=np.linalg.norm(ego_accel, axis=1),  # accel (3D) to speed (1D)
        name_y="Ego Accel",
        units_y="cm/s^2",
        units_x="s",
        lines=True,
    )
    # jerk, snap, crackle, pop?

    # TODO: keep track of vehicles in the scene and track their positions (interpolated) over time

    """steering over time"""
    plot_versus(
        data_x=t,
        name_x="Time",
        data_y=data["UserInputs"]["Steering"],
        name_y="Steering",
        units_y="Deg",
        units_x="s",
        lines=True,
    )

    """velocity over time"""

    plot_vector_vs_time(ego_velocity[50:], t[1 + 50 :], "EgoVel XYZ")
    plot_vector_vs_time(ego_accel[50:], t[2 + 50 :], "EgoAccel XYZ")
    plot_vector_vs_time(data["EgoVariables"]["VehicleRot"], t, "EgoRot XYZ")

    """Stored velocity over time"""
    plot_versus(
        data_x=t,
        name_x="Time",
        data_y=data["EgoVariables"]["VehicleVel"],
        name_y="Velocity",
        units_y="m/s",
        units_x="s",
        lines=True,
    )

    plot_versus(
        data_x=t,
        name_x="Time",
        data_y=data["UserInputs"]["Throttle"],
        name_y="Throttle",
        units_y="",
        units_x="s",
        lines=True,
    )

    plot_versus(
        data_x=t,
        name_x="Time",
        data_y=data["UserInputs"]["Brake"],
        name_y="Brake",
        units_y="",
        units_x="s",
        lines=True,
    )

    """plot relative camera things"""
    plot_vector_vs_time(data["EgoVariables"]["CameraLoc"], t, "CameraLoc")
    plot_vector_vs_time(data["EgoVariables"]["CameraRot"], t, "CameraRot")

    """plot gaze things"""
    plot_vector_vs_time(data["EyeTracker"]["COMBINEDGazeDir"], t, "CombinedGaze")
    plot_vector_vs_time(data["EyeTracker"]["LEFTGazeDir"], t, "LeftGaze")
    plot_vector_vs_time(data["EyeTracker"]["RIGHTGazeDir"], t, "RightGaze")
