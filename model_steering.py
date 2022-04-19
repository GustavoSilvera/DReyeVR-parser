import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn import svm
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from captum.attr import IntegratedGradients
import argparse


"""DReyeVR parser imports"""
from parser import parse_file
from utils import (
    check_for_periph_data,
    fill_gaps,
    filter_to_idxs,
    singleify,
    smooth_arr,
    trim_data,
    flatten_dict,
)
from visualizer import (
    plot_versus,
    plot_vector_vs_time,
    save_figure_to_file,
    set_results_dir,
)

results_dir = "results.model.steering"
set_results_dir(results_dir)

seed = 99
np.random.seed(seed)
torch.manual_seed(seed)

"""Get data"""
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
    np.squeeze(data["EyeTracker"]["LEFTPupilDiameter"]), lambda x: x < 1, mode="mean"
)
data["EyeTracker"]["RIGHTPupilDiameter"] = fill_gaps(
    np.squeeze(data["EyeTracker"]["RIGHTPupilDiameter"]), lambda x: x < 1, mode="mean"
)
# remove all "validity" boolean vectors
for key in list(data["EyeTracker"].keys()):
    if "Valid" in key:
        data["EyeTracker"].pop(key)

# compute ego position derivatives
t = data["TimestampCarla"]["data"] / 1000  # ms to s
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
data["EyeTracker_LEFTGazeDir_1_s"] = smooth_arr(data["EyeTracker_LEFTGazeDir_1"], 20)
data["EyeTracker_LEFTGazeDir_2_s"] = smooth_arr(data["EyeTracker_LEFTGazeDir_2"], 20)
data["EyeTracker_RIGHTGazeDir_1_s"] = smooth_arr(data["EyeTracker_RIGHTGazeDir_1"], 20)
data["EyeTracker_RIGHTGazeDir_2_s"] = smooth_arr(data["EyeTracker_RIGHTGazeDir_2"], 20)
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


"""get data!!!"""
t = data["TimestampCarla_data"] / 1000  # ms to s

Y = data["UserInputs_Steering"]

# Split sampled data into training and test

feature_names = [
    "EgoVariables_VehicleVel",
    "EgoVariables_Velocity_0",  # dependent on steering
    "EgoVariables_Velocity_1",  # dependent on steering
    # "EgoVariables_Velocity_2",  # causes nan's
    # "EgoVariables_AngularVelocity_0",  # ~ should be ~constant (no pitch)
    "EgoVariables_AngularVelocity_1",  # yaw velocity
    # "EgoVariables_AngularVelocity_2",  # should be ~constant (no roll)
    # "EgoVariables_Accel_0", # dependent on steering
    # "EgoVariables_Accel_1", # dependent on steering
    # "EgoVariables_Accel_2",  # uninteresting
    # "EyeTracker_COMBINEDGazeDir_0",  # not interesting, should be ~1
    # "EyeTracker_COMBINEDGazeDir_1_s", # correlated with LEFT/RIGHT
    # "EyeTracker_COMBINEDGazeDir_2_s", # correlated with LEFT/RIGHT
    # "EyeTracker_LEFTGazeDir_0",  # not interesting, should be ~1
    "EyeTracker_LEFTGazeDir_1_s",
    "EyeTracker_LEFTGazeDir_2_s",
    # "EyeTracker_RIGHTGazeDir_0",  # not interesting, should be ~1
    "EyeTracker_RIGHTGazeDir_1_s",
    "EyeTracker_RIGHTGazeDir_2_s",
    "EyeTracker_LEFTPupilDiameter_s",
    "EyeTracker_LEFTPupilPosition_0_s",
    "EyeTracker_LEFTPupilPosition_1_s",
    "EyeTracker_RIGHTPupilDiameter_s",
    "EyeTracker_RIGHTPupilPosition_0_s",
    "EyeTracker_RIGHTPupilPosition_1_s",
    # "EgoVariables_VehicleLoc_0",  # dependent on steering
    # "EgoVariables_VehicleLoc_1",  # dependent on steering
    # "EgoVariables_VehicleLoc_2", # z position mostly flat
    # "EgoVariables_VehicleRot_0", # unwrapped absolute rotators
    # "EgoVariables_VehicleRot_1", # unwrapped absolute rotators
    # "EgoVariables_VehicleRot_2", # unwrapped absolute rotators
    "EgoVariables_CameraLoc_0",
    "EgoVariables_CameraLoc_1",
    # "EgoVariables_CameraLoc_2", # mostly flat
    "EgoVariables_CameraRot_0",  # relative rotators are ok
    "EgoVariables_CameraRot_1",  # relative rotators are ok
    "EgoVariables_CameraRot_2",  # relative rotators are ok
    "UserInputs_Throttle",
    "UserInputs_Brake",
]

X = np.array([data[feature_key] for feature_key in feature_names]).T

# make test/train split
p = 0.2
m = int(len(X) * (1 - p))  # percentage for training
train_split = {"X": X[:m], "Y": Y[:m]}
test_split = {"X": X[m:], "Y": Y[m:]}


class DrivingModel(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = 1  # outputting only a single scalar
        self.network = torch.nn.Sequential(
            nn.Linear(self.in_dim, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, self.out_dim),
        )

    def forward(self, x):
        return self.network(x)


model = DrivingModel(train_split["X"].shape[1])

critereon = torch.nn.MSELoss()
# critereon = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.01)

nb_epochs = 25
acc_thresh = np.mean(np.abs(test_split["Y"]))
for epoch in range(nb_epochs):
    start_t = time.time()
    """train model"""
    model.train()
    train_loss = 0
    for ix, x in enumerate(train_split["X"]):
        optimizer.zero_grad()
        data = torch.Tensor(x)
        desired = torch.Tensor([train_split["Y"][ix]])
        outputs = model.forward(data)
        # predictions = torch.nn.functional.softmax(outputs, dim=1)
        loss = critereon(outputs, desired)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    """test model"""
    test_loss = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for ix, x in enumerate(test_split["X"]):
            data = torch.Tensor(x)
            desired = torch.Tensor([test_split["Y"][ix]])
            outputs = model.forward(data)
            correct += 1 if torch.abs(outputs - desired) < acc_thresh else 0
            loss_crit = critereon(outputs, desired)
            test_loss += loss_crit.item()
        acc = 100 * correct / len(test_split["Y"])
    scheduler.step(test_loss)
    print(
        f"Epoch {epoch} \t Train: {train_loss:4.3f} \t Test: {test_loss:4.3f}"
        f"\t Acc: {acc:2.1f}"
    )

filename: str = os.path.join(results_dir, "model.pt")
torch.save({"steering_net", model.state_dict()}, filename)

# y_pred = np.array([model(torch.Tensor(X[i])).detach().numpy() for i in range(len(X))])
y_pred = model.forward(torch.Tensor(test_split["X"])).detach().numpy()


plot_versus(
    data_x=t[m:],
    data_y=test_split["Y"],
    name_x="Frames",
    name_y="TestY",
    lines=True,
)

plot_versus(
    data_x=t[m:],
    data_y=y_pred,
    name_x="Frames",
    name_y="PredY",
    lines=True,
)

"""test on training data"""
y_pred = np.squeeze(model.forward(torch.Tensor(X)).detach().numpy())
# plot_versus(
#     data_x=t,
#     data_y=y_pred,
#     name_x="Frames",
#     name_y="AllPredY",
#     lines=True,
# )

pred_actual = np.array([y_pred, Y]).T
plot_vector_vs_time(
    xyz=pred_actual, t=t, title="predicted vs actual", ax_titles=["pred", "actual"]
)


def visualize_importances(
    feature_names,
    title="Average Feature Importances",
    axis_title="Features",
):
    print("Visualizing feature importances...")
    # Helper method to print importances and visualize distribution
    ig = IntegratedGradients(model)
    test_input_tensor = torch.Tensor(test_split["X"])
    test_input_tensor.requires_grad_()
    attr, delta = ig.attribute(
        test_input_tensor, target=0, return_convergence_delta=True
    )
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


feature_names_small = [f[f.find("_") + 1 :] for f in feature_names]
visualize_importances(feature_names_small)
