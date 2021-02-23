from WhiteBoxModel import WhiteBoxModel
from data_loader import TestSet
import numpy as np
from torch.utils.data import DataLoader
import torch

def mse_loss(x1, x2):
    # Returns the mean square error loss for two nx1 vectors
    e = np.abs(x1-x2)
    n = x1.shape[0]
    return np.dot(e.T, e)/n


if __name__ == "__main__":
    # Initialize Variables
    l = 0.211  # length (m)
    d = 1.7e-5  # blade parameter
    m = 1  # mass (kg)
    kt = 2.35e-14  # translational drag coefficient
    kr = 0.0099  # rotational drag coefficient
    ixx = 0.002  # moment of inertia about X-axis
    iyy = 0.002  # moment of inertia about Y-axis
    izz = 0.001  # moment of inertia about Z-axis
    lookback = 1
    pred_steps = 60

    test_set = TestSet('data/AscTec_Pelican_Flight_Dataset.mat', lookback, pred_steps, full_set=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

    i = 0
    vel_losses = np.zeros((len(test_set), pred_steps))
    rate_losses = np.zeros((len(test_set), pred_steps))

    # Data initialization
    for data in test_loader:
        input = data["input"][:, 0, :].numpy().T
        # continue here

        model = WhiteBoxModel(l, d, m, kt, kr, ixx, iyy, izz, init_state=input[:12, :])

        for j in range(pred_steps):
            label = data["label"][:, j, :].numpy().T
            model.update_thrust(label[12:, :])
            model.update_torques()
            model.update(0.01)
            vel_loss = mse_loss(model.vel, label[9:12, :])
            rate_loss = mse_loss(model.rate, label[6:9, :])
            vel_losses[i][j] = vel_loss
            rate_losses[i][j] = rate_loss

        i += 1

        if i % 100 == 0:
            print("Iteration: {}".format(i))

    np.savetxt("E2E_v3_multi_test_results_rates.csv", rate_losses)
    np.savetxt("E2E_v3_multi_test_results_vels.csv", vel_losses)

