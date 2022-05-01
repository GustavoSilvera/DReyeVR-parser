import os
import time
import numpy as np
import torch
import argparse


"""DReyeVR parser imports"""
from model_utils import (
    get_model_data,
    get_all_data,
    visualize_importance,
    normalize_batch,
)
from models import DrivingModel
from visualizer import (
    plot_versus,
    plot_vector_vs_time,
    set_results_dir,
)

results_dir = "results.model"
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

"""OUTPUT VARIABLES"""
Y = {}
Y["steering"] = data["UserInputs_Steering"]
Y["throttle"] = data["UserInputs_Throttle"]
Y["brake"] = data["UserInputs_Brake"]

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
]

feature_name_steering = feature_names + [
    "UserInputs_Throttle",  # other driving inputs
    "UserInputs_Brake",  # other driving inputs
]

feature_name_throttle = feature_names + [
    "UserInputs_Steering",  # other driving inputs
    "UserInputs_Brake",  # other driving inputs
]

feature_name_brake = feature_names + [
    "UserInputs_Steering",  # other driving inputs
    "UserInputs_Throttle",  # other driving inputs
]


"""INPUT VARIABLE"""
X = {}
X["steering"] = np.array([data[k] for k in feature_name_steering]).T
X["throttle"] = np.array([data[k] for k in feature_name_throttle]).T
X["brake"] = np.array([data[k] for k in feature_name_brake]).T

# Split sampled data into training and test
p = 0.2  # last 20% of the data is for testing
m = int(len(t) * (1 - p))  # percentage for training
train_split = {
    "X": {k: X[k][:m] for k in X.keys()},
    "Y": {k: Y[k][:m] for k in Y.keys()},
}

test_split = {
    "X": {k: X[k][m:] for k in X.keys()},
    "Y": {k: Y[k][m:] for k in Y.keys()},
}


model = DrivingModel(feature_name_steering, feature_name_throttle, feature_name_brake)
if ckpt is not None:
    assert os.path.exists(ckpt)
    model.load_state_dict(torch.load(ckpt))

model.begin_training(train_split["X"], train_split["Y"], t[:m])

if num_epochs > 0:
    filename: str = os.path.join(results_dir, "model.pt")
    torch.save(model.state_dict(), filename)

model.begin_evaluation(test_split["X"], test_split["Y"], t[m:])

# TODO: compute overall model 
y_pred = model.forward(
    test_split["X"]["steering"],
    test_split["X"]["throttle"],
    test_split["X"]["brake"],
)

# convert back to CPU numpy
steering_prediction = y_pred[0].detach().numpy()
throttle_prediction = y_pred[1].detach().numpy()
brake_prediction = y_pred[2].detach().numpy()
