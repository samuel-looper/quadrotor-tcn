import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from End2EndNet import End2EndNet
from HybridTCN import HybridTCN
from torchdiffeq import odeint
from data_loader import TestSet
from time import perf_counter as count

# neuralnet_eval.py: Evaluate robotic system predictive models over multiple steps


def recurrent_test(test_loader, loss, model, rate_losses, vel_losses, name, device):
    # Evaluates recurrent prediction models on robotic system motion test set
    i = 0
    pred_steps = rate_losses.shape[1]

    with torch.no_grad():
        for data in test_loader:
            raw_input = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2).to(device)  # Load Input data
            label = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2).to(device)      # Load labels
            pred = torch.zeros((bs, 16, 1))     # Initialize prediction tensor

            # Iterate through all timesteps sequentially
            for j in range(pred_steps):
                output_gt = label[0, 6:12, j]           # Ground truth label for that timestep
                feedforward = torch.zeros((bs, 16, 1))  # Future control inputs for that timestep
                feedforward[:, -4:, 0] = label[:, -4:, j]

                # Generate fixed set of past states on a rolling basis for that timestep
                if j == 0:
                    input = torch.cat((raw_input, feedforward), 2)
                else:
                    next_timestep = torch.zeros((bs, 16, 1))
                    next_timestep[0, 0:12, 0] = pred[:12]
                    next_timestep[0, 12:, 0] = input[0, 12:, -1]
                    input = torch.cat((input[:, :, 1:-1], next_timestep, feedforward), 2)

                output = odeint(model, input, torch.tensor([0, 0.01]))  # Prediction for next timestep

                # Calculate state prediction loss
                pred = output[1, 0, :, -2]
                rate_loss = loss(pred[6:9], output_gt[:3])
                vel_loss = loss(pred[9:12], output_gt[3:])
                rate_losses[i, j] = vel_loss
                vel_losses[i, j] = rate_loss

            i += 1
            print("Sample #{}".format(i))

        # Store results in CSV
        np.savetxt("{}_test_error_rates.csv".format(name), rate_losses.cpu().numpy())
        np.savetxt("{}_test_results_vels.csv".format(name), vel_losses.cpu().numpy())


def conv_test(test_loader, loss, model, rate_losses, vel_losses, name, device):
    # Evaluates convolutional (i.e. one-shot) prediction models on robotic system motion test set
    i = 0
    time = []
    pred_steps = rate_losses.shape[1]

    with torch.no_grad():
        for data in test_loader:
            input = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2).to(device)  # Load Input data
            label = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2).to(device)  # Load labels

            output = label[:, 6:12, :]                  # Ground truth label for truncated state
            feedforward = torch.zeros(label.shape)      # Future control inputs
            feedforward[:, 12:, :] = label[:, 12:, :]
            input = torch.cat((input, feedforward), 2)  # Full model input

            start = count()                             # Timing forward pass
            pred = model(input)                         # Future state predictions
            end = count()
            time.append(end-start)

            # Calculate prediction losses
            for j in range(pred_steps):
                rate_loss = loss(output[0, :3, j], pred[0, :3, j]).item()
                vel_loss = loss(output[0, 3:, j], pred[0, 3:, j]).item()

                vel_losses[i][j] = vel_loss
                rate_losses[i][j] = rate_loss

            i += 1
            if i % 10 == 0:
                print("Sample #{}".format(i))

        # Store results in CSV
        print("Average processing time: {} seconds".format(np.asarray(time).mean()))
        np.savetxt("{}_test_error_rates.csv".format(name), rate_losses.cpu().numpy())
        np.savetxt("{}_test_error_vels.csv".format(name), vel_losses.cpu().numpy())


if __name__ == "__main__":
    # Simulation Model Parameters
    l = 0.211  # length (m)
    d = 1.7e-5  # blade parameter
    m = 1  # mass (kg)
    kt = 2.35e-14  # translational drag coefficient
    kr = 0.0099  # rotational drag coefficient
    ixx = 0.002  # moment of inertia about X-axis
    iyy = 0.002  # moment of inertia about Y-axis
    izz = 0.001  # moment of inertia about Z-axis

    # Hyperparameters
    lr = 0.0001
    wd = 0.00005
    epochs = 50
    bs = 1
    P = 64
    F = 90
    loss = nn.L1Loss()

    # Define training/validation datasets and dataloaders
    test_set = TestSet('data/AscTec_Pelican_Flight_Dataset.mat', P, F, full_state=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=True, num_workers=0)
    n = int(len(test_set) / bs)
    print("Testing Length: {}".format(n))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print("GPU")
    else:
        device = torch.device("cpu")
        print("CPU")

    # Main Evaluation Loop
    name = "End2End"
    model = End2EndNet(P, F)
    model.load_state_dict(torch.load('./{}.pth'.format(name), map_location=device))
    model.train(False)
    model.eval()
    vel_losses_E2E = torch.zeros((n, F))
    rate_losses_E2E = torch.zeros((n, F))
    conv_test(test_loader, loss, model, rate_losses_E2E, vel_losses_E2E, name, device)

    name = "MotorHybrid"
    model = HybridTCN(l, m, d, kt, kr, ixx, iyy, izz, P, device, motor=True, accel_error=False)
    model.load_state_dict(torch.load('./{}.pth'.format(name), map_location=device))
    model.train(False)
    model.eval()
    vel_losses_MH = torch.zeros((n, F))
    rate_losses_MH = torch.zeros((n, F))
    recurrent_test(test_loader, loss, model, rate_losses_MH, vel_losses_MH, name, device)





