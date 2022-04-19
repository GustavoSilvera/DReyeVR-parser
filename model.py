import time
import numpy as np
import pandas as pd
import torch
from sklearn import svm
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import argparse


"""DReyeVR parser imports"""
from parser import parse_file
from utils import (
    check_for_periph_data,
    fill_gaps,
    filter_to_idxs,
    trim_data,
)
from visualizer import plot_versus, plot_vector_vs_time, set_results_dir

set_results_dir("results.model")

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
ego_accel = np.concatenate((np.zeros((2, 3)), ego_accel))
data["EgoVariables"]["Velocity"] = ego_velocity
# data["EgoVariables"]["Accel"] = np.array([0, 0] + ego_accel.tolist())

# trim data bounds
data = trim_data(data, (50, 100))
t = data["TimestampCarla"]["data"] / 1000  # ms to s

Y = data["UserInputs"]["Steering"]

# Split sampled data into training and test
X = np.array(
    [
        data["EgoVariables"]["VehicleVel"],
        data["EgoVariables"]["Velocity"][:, 0],
        data["EgoVariables"]["Velocity"][:, 1],
        # data["EgoVariables"]["Velocity"][:, 2], # causes nan's
        # data["EyeTracker"]["COMBINEDGazeDir"][:, 0], # highly discrete, should be ~1
        data["EyeTracker"]["COMBINEDGazeDir"][:, 1],
        data["EyeTracker"]["COMBINEDGazeDir"][:, 2],
        # data["EyeTracker"]["LEFTGazeDir"][:, 0], # highly discrete, should be ~1
        data["EyeTracker"]["LEFTGazeDir"][:, 1],
        data["EyeTracker"]["LEFTGazeDir"][:, 2],
        # data["EyeTracker"]["RIGHTGazeDir"][:, 0], # highly discrete, should be ~1
        data["EyeTracker"]["RIGHTGazeDir"][:, 1],
        data["EyeTracker"]["RIGHTGazeDir"][:, 2],
        data["EyeTracker"]["LEFTPupilDiameter"],
        data["EyeTracker"]["LEFTPupilPosition"][:, 0],
        data["EyeTracker"]["LEFTPupilPosition"][:, 1],
        data["EyeTracker"]["RIGHTPupilDiameter"],
        data["EyeTracker"]["RIGHTPupilPosition"][:, 0],
        data["EyeTracker"]["RIGHTPupilPosition"][:, 1],
        data["EgoVariables"]["VehicleLoc"][:, 0],
        data["EgoVariables"]["VehicleLoc"][:, 1],
        # data["EgoVariables"]["VehicleLoc"][:, 2], # z position is mostly flat
        # data["EgoVariables"]["VehicleRot"][:, 0], # rotators are just weird (non wrapped)
        # data["EgoVariables"]["VehicleRot"][:, 1], # rotators are just weird (non wrapped)
        # data["EgoVariables"]["VehicleRot"][:, 2], # rotators are just weird (non wrapped)
        data["EgoVariables"]["CameraLoc"][:, 0],
        data["EgoVariables"]["CameraLoc"][:, 1],
        # data["EgoVariables"]["CameraLoc"][:, 2], # z position is mostly flat
        data["EgoVariables"]["CameraRot"][:, 0],  # relative rotators are ok
        data["EgoVariables"]["CameraRot"][:, 1],  # relative rotators are ok
        data["EgoVariables"]["CameraRot"][:, 2],  # relative rotators are ok
        data["UserInputs"]["Throttle"],
        # data["UserInputs"]["Steering"],
        data["UserInputs"]["Brake"],
    ]
).T

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

nb_epochs = 20
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
