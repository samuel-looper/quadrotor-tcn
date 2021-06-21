import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
NUM_TRAIN = 1388302-34709+2  # Precomputed number of training samples
NUM_TEST = 34709             # Precomputed number of testing samples


class WBDataLoader:
    # Data loader for simple physics-based "white box" model
    def __init__(self, filepath):
        self.data = sio.loadmat(filepath)
        self.f_len = 0
        self.f_pos = np.zeros((3, 30000))
        self.f_ang = np.zeros((3, 30000))
        self.f_motor_speed = np.zeros((4, 30000))
        self.f_motor_cmd = np.zeros((4, 30000))
        self.f_vel = np.zeros((3, 30000))
        self.f_rate = np.zeros((3, 30000))

    # Loads full state telemetry a single flight from self.data as a series of numpy arrays
    def get_flight(self, i):
        flight = self.data["flights"][0, i]
        self.f_len = flight["len"][0, 0][0][0]          # Length of flight
        self.f_pos = flight["Pos"][0, 0]                # (x,y,z) position measurements
        self.f_ang = flight["Euler"][0, 0]              # Euler angle (pitch, roll, yaw) measurements in deg.
        self.f_motor_speed = flight["Motors"][0, 0]     # Motor Speeds (0 to 218 integer value)
        self.f_motor_cmd = flight["Motors_CMD"][0, 0]   # Commands sent to quadrotor
        self.f_vel = flight["Vel"][0, 0]                # Position velocity over time (x, y, z)
        self.f_rate = flight["pqr"][0, 0]               # Euler body rates (p, q, r) in deg.

        return 0


class TrainSet(Dataset):
    # Generates input and label sequences from a quadrotor telemetry dataset for a machine learning training set.
    def __init__(self, filepath, input_size, output_size, full_set=False):
        if full_set:
            chan = 16
        else:
            chan = 10
        self.data = sio.loadmat(filepath)
        self.scale_factor = 1
        size = np.floor(NUM_TRAIN / max(input_size, output_size)).astype(int)
        self.inputs = np.zeros((size*self.scale_factor, input_size, chan))
        self.outputs = np.zeros((size*self.scale_factor, output_size, chan))
        ind = 0
        # Collect all flight data from Matlab matrices
        for count, flight in enumerate(self.data["flights"][0, :]):
            if count != 18:
                f_pos = flight["Pos"][0, 0]  # (x,y,z) position measurements
                f_ang = flight["Euler"][0, 0]  # Euler angle (pitch, roll, yaw) measurements in deg.
                f_motor_cmd = flight["Motors_CMD"][0, 0]  # Commands sent to quadrotor
                f_vel = flight["Vel"][0, 0]  # Position velocity over time (x, y, z)
                f_rate = flight["pqr"][0, 0]
                if input_size > output_size:
                    length = np.floor(f_vel.shape[0] / input_size).astype(int) * input_size - input_size
                    interval = int(length / input_size)
                    step = np.floor(input_size / self.scale_factor).astype(int)
                    end = step * self.scale_factor
                    for offset in range(0, end, step):
                        if full_set:
                            state = np.hstack((f_ang[1+offset:length + 1 + offset, :], f_pos[1+ offset:length + 1+ offset, :], f_rate[1+ offset:length + 1+ offset, :], f_vel[offset:length+ offset, :], f_motor_cmd[1+ offset:length + 1+ offset, :]))
                        else:
                            state = np.hstack((f_rate[1+offset:length+1+offset, :], f_vel[offset:length+offset, :], f_motor_cmd[1+offset:length+offset+1, :]))
                        state = np.reshape(state, (interval, input_size, chan))

                        output_state = state[1:, :output_size, :]
                        input_state = state[:-1, :, :]
                        self.inputs[ind:ind+interval-1, :, :] = input_state
                        self.outputs[ind:ind+interval-1, :, :] = output_state
                        ind += interval-1
                else:
                    length = np.floor(f_vel.shape[0] / output_size).astype(int) * output_size - output_size
                    step = np.floor(output_size / self.scale_factor).astype(int)
                    end = step * self.scale_factor
                    for offset in range(0, end, step):
                        if full_set:
                            state = np.hstack((f_ang[1+offset:length + 1 +offset, :], f_pos[1+offset:length + 1+offset, :], f_rate[1+offset:length + 1+offset, :], f_vel[offset:length+offset, :], f_motor_cmd[1+offset:length + 1+offset, :]))
                        else:
                            state = np.hstack((f_rate[1:length+1+offset, :], f_vel[offset:length+offset, :], f_motor_cmd[1+offset:length+offset+1, :]))
                        interval = int(length/output_size)
                        state = np.reshape(state, (interval, output_size, chan))
                        output_state = state[1:, :, :]
                        input_state = state[:-1, -input_size:, :]
                        self.inputs[ind:ind + interval - 1, :, :] = input_state
                        self.outputs[ind:ind + interval - 1, :, :] = output_state
                        ind += interval - 1
                # print(count)
        self.inputs = self.inputs[:ind, :, :]
        self.outputs = self.outputs[:ind, :, :]
        # Normalize motor commands
        self.normalize_commands()

    def normalize_everything(self):
        mean = np.mean(self.inputs, axis=(0, 1))
        std = np.std(self.inputs, axis=(0, 1))
        self.inputs = (self.inputs - mean) / std
        self.outputs = (self.outputs - mean) / std

    def normalize_commands(self):
        mean = np.mean(self.inputs[:, :, -4:], axis=(0, 1))
        std = np.std(self.inputs[:, :, -4:], axis=(0, 1))
        self.inputs[:, :, -4:] = (self.inputs[:, :, -4:] - mean) / std
        self.outputs[:, :, -4:] = (self.outputs[:, :, -4:] - mean) / std


    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'input': self.inputs[idx, :, :], 'label': self.outputs[idx, :, :]}

        return sample


class TestSet(Dataset):
    def __init__(self, filepath, input_size, output_size, full_set=False):
        if full_set:
            chan = 16
        else:
            chan = 10
        self.data = sio.loadmat(filepath)
        size = np.floor(NUM_TEST / input_size).astype(int)
        self.inputs = np.zeros((size, input_size, chan))
        self.outputs = np.zeros((size, output_size, chan))
        ind = 0
        # Collect all flight data from Matlab matrices
        for count, flight in enumerate(self.data["flights"][0, :]):
            if count == 18:
                f_pos = flight["Pos"][0, 0]  # (x,y,z) position measurements
                f_ang = flight["Euler"][0, 0]  # Euler angle (pitch, roll, yaw) measurements in deg.
                f_motor_cmd = flight["Motors_CMD"][0, 0]  # Commands sent to quadrotor
                f_vel = flight["Vel"][0, 0]  # Position velocity over time (x, y, z)
                f_rate = flight["pqr"][0, 0]
                if input_size > output_size:
                    length = np.floor(f_vel.shape[0] / input_size).astype(int) * input_size
                    if full_set:
                        state = np.hstack((f_ang[1:length + 1, :], f_pos[1:length + 1, :], f_rate[1:length + 1, :],
                                           f_vel[:length, :], f_motor_cmd[1:length + 1, :]))
                    else:
                        state = np.hstack((f_rate[1:length + 1, :], f_vel[:length, :], f_motor_cmd[1:length + 1, :]))
                    interval = int(length / input_size)
                    state = np.reshape(state, (interval, input_size, chan))

                    output_state = state[1:, :output_size, :]
                    input_state = state[:-1, :, :]
                    self.inputs[ind:ind + interval - 1, :, :] = input_state
                    self.outputs[ind:ind + interval - 1, :, :] = output_state
                    ind += interval - 1
                else:
                    length = np.floor(f_vel.shape[0] / output_size).astype(int) * output_size
                    if full_set:
                        state = np.hstack((f_ang[1:length + 1, :], f_pos[1:length + 1, :], f_rate[1:length + 1, :],
                                           f_vel[:length, :], f_motor_cmd[1:length + 1, :]))
                    else:
                        state = np.hstack((f_rate[1:length + 1, :], f_vel[:length, :], f_motor_cmd[1:length + 1, :]))
                    interval = int(length / output_size)
                    state = np.reshape(state, (interval, output_size, chan))
                    output_state = state[1:, :, :]
                    input_state = state[:-1, -input_size:, :]
                    self.inputs[ind:ind + interval - 1, :, :] = input_state
                    self.outputs[ind:ind + interval - 1, :, :] = output_state
                    ind += interval - 1
        self.inputs = self.inputs[:ind, :, :]
        self.outputs = self.outputs[:ind, :, :]
        # Normalize motor commands
        self.normalize_commands()

    def normalize_everything(self):
        mean = np.mean(self.inputs, axis=(0, 1))
        std = np.std(self.inputs, axis=(0, 1))
        self.inputs = (self.inputs - mean) / std
        self.outputs = (self.outputs - mean) / std

    def normalize_commands(self):
        mean = np.mean(self.inputs[:, :, -4:], axis=(0, 1))
        std = np.std(self.inputs[:, :, -4:], axis=(0, 1))
        self.inputs[:, :, -4:] = (self.inputs[:, :, -4:] - mean) / std
        self.outputs[:, :, -4:] = (self.outputs[:, :, -4:] - mean) / std

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'input': self.inputs[idx, :, :], 'label': self.outputs[idx, :, :]}

        return sample

    # Test Comment