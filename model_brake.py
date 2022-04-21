import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn import svm
from torch.utils.data import DataLoader, TensorDataset
import argparse


"""DReyeVR parser imports"""
from model_utils import (
    get_model_data,
    get_all_data,
    visualize_importance,
    normalize_batch,
)
from models import BrakeModel
from visualizer import (
    plot_versus,
    plot_vector_vs_time,
    set_results_dir,
)

results_dir = "results.model.brake"
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
argparser.add_argument(
    "--load",
    metavar="L",
    default=None,
    type=str,
    help="path to a saved model state dict checkpoint",
)
argparser.add_argument(
    "--epochs",
    metavar="E",
    default=0,
    type=int,
    help="Number of epochs to train this model",
)
args = argparser.parse_args()
filename: str = args.file
ckpt: str = args.load
num_epochs = args.epochs

if filename is None:
    print("Need to pass in the recording file")
    exit(1)

# data = get_model_data(filename)
data = get_all_data(filename)
data = normalize_batch(data)

"""get data!!!"""
t = data["TimestampCarla_data"]

"""OUTPUT VARIABLE"""
Y = data["UserInputs_Brake"]

feature_names = [
    "EgoVariables_VehicleLoc_0",
    "EgoVariables_VehicleLoc_1",
    "EgoVariables_VehicleVel",
    "EgoVariables_Velocity_0",
    "EgoVariables_Velocity_1",
    "EgoVariables_AngularVelocity_1",  # yaw velocity
    "EyeTracker_LEFTGazeDir_1_s",
    "EyeTracker_LEFTGazeDir_2_s",
    "EyeTracker_RIGHTGazeDir_1_s",
    "EyeTracker_RIGHTGazeDir_2_s",
    "EyeTracker_LEFTPupilDiameter_s",
    "EyeTracker_LEFTPupilPosition_0_s",
    "EyeTracker_LEFTPupilPosition_1_s",
    "EyeTracker_RIGHTPupilDiameter_s",
    "EyeTracker_RIGHTPupilPosition_0_s",
    "EyeTracker_RIGHTPupilPosition_1_s",
    "EgoVariables_CameraLoc_0",
    "EgoVariables_CameraLoc_1",
    "EgoVariables_CameraRot_0",
    "EgoVariables_CameraRot_1",
    "UserInputs_Steering",  # other driving inputs
    "UserInputs_Throttle",  # other driving inputs
]

"""INPUT VARIABLE"""
X = np.array([data[feature_key] for feature_key in feature_names]).T

# Split sampled data into training and test
# make test/train split
p = 0.2
m = int(len(X) * (1 - p))  # percentage for training
train_split = {"X": X[:m], "Y": Y[:m]}
test_split = {"X": X[m:], "Y": Y[m:]}


model = BrakeModel(len(feature_names))
if ckpt is not None:
    assert os.path.exists(ckpt)
    model.load_state_dict(torch.load(ckpt))


critereon = torch.nn.L1Loss()  # more robust than MSE for outliers
# critereon = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.01)

print("Starting model training...")
acc_thresh = np.mean(np.abs(test_split["Y"]))
accs = []
losses = []
for epoch in range(num_epochs):
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
        accs.append(acc)
        losses.append(test_loss)
    scheduler.step(test_loss + train_loss)
    print(
        f"Epoch {epoch} \t Train: {train_loss:4.3f} \t Test: {test_loss:4.3f}"
        f"\t Acc: {acc:2.1f}"
    )
    full_predictions = np.array(
        [np.squeeze(model.forward(torch.Tensor(X)).detach().numpy()), Y]
    ).T
    plot_vector_vs_time(
        xyz=full_predictions,
        t=t,
        title=f"predicted vs actual.{epoch}",
        ax_titles=["pred", "actual"],
        silent=True,
    )


if num_epochs > 0:
    filename: str = os.path.join(results_dir, "model.pt")
    torch.save(model.state_dict(), filename)

# use for inference now
model.eval()

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

# plot accuracy
assert len(accs) == num_epochs
plot_versus(
    data_x=np.arange(num_epochs),
    data_y=accs,
    name_x="Epochs",
    name_y="Accuracy",
    lines=True,
)

# plot losses
assert len(losses) == num_epochs
plot_versus(
    data_x=np.arange(num_epochs),
    data_y=losses,
    name_x="Epochs",
    name_y="Loss",
    lines=True,
)

"""test on training data"""
y_pred = np.squeeze(model.forward(torch.Tensor(X)).detach().numpy())

pred_actual = np.array(
    [np.squeeze(model.forward(torch.Tensor(X)).detach().numpy()), Y]
).T
plot_vector_vs_time(
    xyz=pred_actual, t=t, title="predicted vs actual", ax_titles=["pred", "actual"]
)


feature_names_small = [f[f.find("_") + 1 :] for f in feature_names]
visualize_importance(model, feature_names_small, torch.Tensor(test_split["X"]))
