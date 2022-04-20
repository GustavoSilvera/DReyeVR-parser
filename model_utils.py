import os
import numpy as np
from typing import Dict, Any, Optional
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import pickle

"""parser imports & utils"""
from parser import parse_file
from utils import (
    check_for_periph_data,
    filter_to_idxs,
    fill_gaps,
    trim_data,
    flatten_dict,
    singleify,
    smooth_arr,
)
from visualizer import (
    save_figure_to_file,
)

data_dir = "data"


def get_model_data(filename: str) -> Dict[str, Any]:
    """try to load cached data from previous runs"""
    data = try_load_data(filename)
    if data is not None:
        return data

    """Have to read the file to create the object"""
    data = parse_file(filename)
    # check for periph data
    PeriphData = check_for_periph_data(data)
    if PeriphData is not None:
        data["PeriphData"] = PeriphData

    """sanitize data"""
    if "CustomActor" in data:
        data.pop("CustomActor")  # not using this rn
    data = filter_to_idxs(data, mode="all")
    data["EyeTracker"]["LEFTPupilDiameter"] = fill_gaps(
        np.squeeze(data["EyeTracker"]["LEFTPupilDiameter"]),
        lambda x: x < 1,
        mode="mean",
    )
    data["EyeTracker"]["RIGHTPupilDiameter"] = fill_gaps(
        np.squeeze(data["EyeTracker"]["RIGHTPupilDiameter"]),
        lambda x: x < 1,
        mode="mean",
    )
    # remove all "validity" boolean vectors
    for key in list(data["EyeTracker"].keys()):
        if "Valid" in key:
            data["EyeTracker"].pop(key)

    # compute ego position derivatives
    t = data["TimestampCarla"]["data"] / 1000  # ms to s
    data["TimestampCarla"]["data"] = t
    delta_ts = np.diff(t)  # t is in seconds
    n: int = len(delta_ts)
    assert delta_ts.min() > 0  # should always be monotonically increasing!
    ego_displacement = np.diff(data["EgoVariables"]["VehicleLoc"], axis=0)
    ego_velocity = (ego_displacement.T / delta_ts).T
    ego_velocity = np.concatenate((np.zeros((1, 3)), ego_velocity))  # include 0 @ t=0
    ego_accel = (np.diff(ego_velocity, axis=0).T / delta_ts).T
    ego_accel = np.concatenate((np.zeros((1, 3)), ego_accel))  # include 0 @ t=0
    data["EgoVariables"]["Velocity"] = ego_velocity
    data["EgoVariables"]["Accel"] = ego_accel
    rot3D = data["EgoVariables"]["VehicleRot"]
    angular_disp = np.diff(rot3D, axis=0)
    # fix rollovers for +360
    angular_disp[
        np.squeeze(np.where(np.abs(np.diff(rot3D[:, 1], axis=0)) > 359))
    ] = 0  # TODO
    # pos_roll_idxs = np.squeeze(np.where(np.diff(rot3D[:, 1], axis=0) > 359))
    # angular_disp[pos_roll_idxs][:, 1] = -1 * (360 - angular_disp[pos_roll_idxs][:, 1])
    # neg_roll_idxs = np.squeeze(np.where(np.diff(rot3D[:, 1], axis=0) < -359))
    # angular_disp[neg_roll_idxs][:, 1] = 360 + angular_disp[neg_roll_idxs][:, 1]
    angular_vel = (angular_disp.T / delta_ts).T
    angular_vel = np.concatenate((np.zeros((1, 3)), angular_disp))  # include 0 @ t=0
    data["EgoVariables"]["AngularVelocity"] = angular_vel

    # trim data bounds
    data = trim_data(data, (50, 100))
    data = flatten_dict(data)
    data = singleify(data)  # so individual axes are accessible via _ notation

    # apply data smoothing
    data["EyeTracker_COMBINEDGazeDir_1_s"] = smooth_arr(
        data["EyeTracker_COMBINEDGazeDir_1"], 20
    )
    data["EyeTracker_COMBINEDGazeDir_2_s"] = smooth_arr(
        data["EyeTracker_COMBINEDGazeDir_2"], 20
    )
    data["EyeTracker_LEFTGazeDir_1_s"] = smooth_arr(
        data["EyeTracker_LEFTGazeDir_1"], 20
    )
    data["EyeTracker_LEFTGazeDir_2_s"] = smooth_arr(
        data["EyeTracker_LEFTGazeDir_2"], 20
    )
    data["EyeTracker_RIGHTGazeDir_1_s"] = smooth_arr(
        data["EyeTracker_RIGHTGazeDir_1"], 20
    )
    data["EyeTracker_RIGHTGazeDir_2_s"] = smooth_arr(
        data["EyeTracker_RIGHTGazeDir_2"], 20
    )
    data["EyeTracker_LEFTPupilDiameter_s"] = smooth_arr(
        data["EyeTracker_LEFTPupilDiameter"], 20
    )
    data["EyeTracker_LEFTPupilPosition_0_s"] = smooth_arr(
        data["EyeTracker_LEFTPupilPosition_0"], 20
    )
    data["EyeTracker_LEFTPupilPosition_1_s"] = smooth_arr(
        data["EyeTracker_LEFTPupilPosition_1"], 20
    )
    data["EyeTracker_RIGHTPupilDiameter_s"] = smooth_arr(
        data["EyeTracker_RIGHTPupilDiameter"], 20
    )
    data["EyeTracker_RIGHTPupilPosition_0_s"] = smooth_arr(
        data["EyeTracker_RIGHTPupilPosition_0"], 20
    )
    data["EyeTracker_RIGHTPupilPosition_1_s"] = smooth_arr(
        data["EyeTracker_RIGHTPupilPosition_1"], 20
    )
    cache_data(data, filename)
    return data


def visualize_importance(
    model,
    feature_names,
    input_tensor,
    title="Average Feature Importances",
    axis_title="Features",
):
    print("Visualizing feature importances...")
    # Helper method to print importances and visualize distribution
    ig = IntegratedGradients(model)
    input_tensor.requires_grad_()
    attr, delta = ig.attribute(input_tensor, target=0, return_convergence_delta=True)
    attr = attr.detach().numpy()
    importances = np.mean(attr, axis=0) / np.abs(np.mean(attr))
    for i in range(len(feature_names)):
        print(f"{feature_names[i]} : {importances[i]:.3f}")
    x_pos = np.arange(len(feature_names))

    fig = plt.figure(figsize=(12, 8))
    plt.grid(True)
    plt.bar(x_pos, importances, align="center")
    plt.xticks(x_pos, feature_names, wrap=True, rotation=80)
    plt.xlabel(axis_title)
    plt.title(title)
    save_figure_to_file(fig, "feature_importance.png")


def try_load_data(filename) -> Optional[Dict[str, Any]]:
    actual_name = filename.split("/")[-1].replace(".txt", "")
    filename = f"{os.path.join(data_dir, actual_name)}.data"
    data = None
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded data from {filename}")
    else:
        print(f"Did not find data at {filename}")
    return data


def cache_data(data, filename):
    actual_name: str = filename.split("/")[-1].replace(".txt", "")
    os.makedirs(data_dir, exist_ok=True)
    filename = f"{os.path.join(data_dir, actual_name)}.data"
    with open(filename, "wb") as filehandler:
        pickle.dump(data, filehandler)
    print(f"cached data to {filename}")
