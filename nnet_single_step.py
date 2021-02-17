import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import SysIDModel
import MotorHybridNet
import AccelErrorNet
import FullHybridNet
import End2EndNet
from torchdiffeq import odeint
from data_loader import SinglePredDatasetTest


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

    lookback = 64
    pred_steps = 1
    bs = 1


    # Data initialization
    test_set = SinglePredDatasetTest('data/AscTec_Pelican_Flight_Dataset.mat', lookback, pred_steps, full_set=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=True, num_workers=0)

    full_error_rec = torch.zeros((len(test_set), 1))
    accel_error_rec = torch.zeros((len(test_set), 1))
    state_rec = torch.zeros((len(test_set), 16))

    model = QuadrotorDynamics()
    model.load_state_dict(torch.load('./E2E_1Step_best.pth'))
    model.eval()
    loss_f = nn.MSELoss()
    i = 0
    print("Testing Length: {}".format(int(len(test_set) / bs)))
    with torch.no_grad():
        for data in test_loader:
            raw_input = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2)    # Load Input data
            label = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2)        # Load labels
            output_gt = label[0, :, 0]
            feedforward = torch.zeros(label.shape)
            feedforward[:, -4:, :] = label[:, -4:, :]
            input = torch.cat((raw_input, feedforward), 2)

            output = odeint(model, input, torch.tensor([0, 0.01]))[1, 0, :, -2]

            full_loss = loss_f(output[:12], output_gt[:12])
            full_error_rec[i, 0] = full_loss

            accel_loss = loss_f(output[6:12], output_gt[6:12])
            accel_error_rec[i, 0] = accel_loss

            state_rec[i, :] = output_gt

            if i % 100 == 0:
                print("Interation: {}".format(i))
                print("Full State Error: {} \n".format(full_error_rec[i, 0]))

            i += 1


    print("before plot")
    n, bins, patches = plt.hist(full_error_rec.numpy(), 50, density=False, facecolor='g', alpha=0.75)
    plt.title('Full State Error Distribution')
    plt.savefig("full_state_error.png")
    plt.show()

    n, bins, patches = plt.hist(accel_error_rec.numpy(), 50, density=False, facecolor='g', alpha=0.75)
    plt.title('Velocity & Body Rate Error Distribution')
    plt.savefig("velocity_error.png")
    plt.show()

    full_rec = torch.cat((state_rec, full_error_rec, accel_error_rec), 1)
    np.savetxt("full_hybrid_test_results.csv", full_rec.numpy())




