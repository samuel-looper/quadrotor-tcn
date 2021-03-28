import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from End2EndNet import E2ESingleStepTCNv3, E2ESingleStepTCNv4, E2ESingleStepTCNv5, E2ESingleStepTCNv6
from torchdiffeq import odeint
from data_loader import TestSet


def recurrent_test(test_loader, model, rate_losses, vel_losses):
    loss_f = nn.L1Loss()
    i = 0
    j = 0

    with torch.no_grad():
        for data in test_loader:
            raw_input = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2)  # Load Input data
            label = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2)  # Load labels
            # label_n = label.numpy()
            pred = torch.zeros((bs, 16, 1))
            for j in range(pred_steps):
                output_gt = label[0, 6:12, j] # Get GT for that timestep
                if j == 0:
                    feedforward = torch.zeros((bs, 16, 1))
                    feedforward[:, -4:, 0] = label[:, -4:, j]
                    input = torch.cat((raw_input, feedforward), 2)
                else:
                    feedforward = torch.zeros((bs, 16, 1))
                    feedforward[:, -4:, 0] = label[:, -4:, j]
                    next_timestep = torch.zeros((bs, 16, 1))
                    next_timestep[0, 0:12, 0] = pred[:12]
                    next_timestep[0, 12:, 0] = input[0, 12:, -1]
                    input = torch.cat((input[:, :, 1:-1], next_timestep, feedforward), 2)

                output = odeint(model, input, torch.tensor([0, 0.01]))
                pred = output[1, 0, :, -2]
                rate_loss = loss_f(pred[6:9], output_gt[:3])
                vel_loss = loss_f(pred[9:12], output_gt[3:])
                rate_losses[i, j] = vel_loss
                vel_losses[i, j] = rate_loss

            i += 1
            if i % 10 == 0:
                print("Sample #{}".format(i))

        np.savetxt("MH_v3_multi_test_results_rates.csv", rate_losses.numpy())
        np.savetxt("MH_v3_multi_test_results_vels.csv", vel_losses.numpy())


def conv_test(test_loader, net, rate_losses, vel_losses, name):
    loss_f = nn.MSELoss()
    i = 0
    with torch.no_grad():
        for data in test_loader:
            j = 0
            input = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2)  # Load Input data
            label = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2)  # Load labels

            output = label[:, 6:12, :]
            feedforward = torch.zeros(label.shape)
            feedforward[:, 12:, :] = label[:, 12:, :]
            input = torch.cat((input, feedforward), 2)

            pred = net(input)
            for j in range(pred_steps):
                rate_loss = loss_f(output[0, :3, j], pred[0, :3, j]).item()
                vel_loss = loss_f(output[0, 3:, j], pred[0, 3:, j]).item()

                vel_losses[i][j] = vel_loss
                rate_losses[i][j] = rate_loss

            i += 1
            if i % 10 == 0:
                print("Sample #{}".format(i))

        np.savetxt("{}_test_error_rates.csv".format(name), rate_losses.numpy())
        np.savetxt("{}_test_error_vels.csv".format(name), vel_losses.numpy())


if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # torch.set_default_tensor_type("torch.cuda.FloatTensor")

    lookback = 64
    pred_steps = 90
    bs = 1

    # Data initialization
    test_set = TestSet('data/AscTec_Pelican_Flight_Dataset.mat', lookback, pred_steps, full_set=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=True, num_workers=0)
    n = int(len(test_set) / bs)
    print("Testing Length: {}".format(n))


    name = "E2E_v3"
    model = E2ESingleStepTCNv3(lookback, pred_steps)
    model.load_state_dict(torch.load('./{}.pth'.format(name), map_location=torch.device("cpu")))
    model.train(False)
    model.eval()

    vel_losses_v3 = torch.zeros((n, pred_steps))
    rate_losses_v3 = torch.zeros((n, pred_steps))
    conv_test(test_loader, model, rate_losses_v3, vel_losses_v3, name)

    name = "E2E_v4"
    model = E2ESingleStepTCNv4(lookback, pred_steps)
    model.load_state_dict(torch.load('./{}.pth'.format(name), map_location=torch.device("cpu")))
    model.train(False)
    model.eval()

    vel_losses_v4 = torch.zeros((n, pred_steps))
    rate_losses_v4 = torch.zeros((n, pred_steps))
    conv_test(test_loader, model, rate_losses_v4, vel_losses_v4, name)

    name = "E2E_v5"
    model = E2ESingleStepTCNv5(lookback, pred_steps)
    model.load_state_dict(torch.load('./{}.pth'.format(name), map_location=torch.device("cpu")))
    model.train(False)
    model.eval()

    vel_losses_v5 = torch.zeros((n, pred_steps))
    rate_losses_v5 = torch.zeros((n, pred_steps))
    conv_test(test_loader, model, rate_losses_v5, vel_losses_v5, name)

    name = "E2E_v6"
    model = E2ESingleStepTCNv6(lookback, pred_steps)
    model.load_state_dict(torch.load('./{}.pth'.format(name), map_location=torch.device("cpu")))
    model.train(False)
    model.eval()

    vel_losses_v6 = torch.zeros((n, pred_steps))
    rate_losses_v6 = torch.zeros((n, pred_steps))
    conv_test(test_loader, model, rate_losses_v6, vel_losses_v6, name)

    time = np.arange(0, 90*10, 10)
    fig1, ax1 = plt.subplots()
    ax1.plot(time, np.mean(vel_losses_v3.numpy(), axis=0), linewidth=2)
    ax1.plot(time, np.mean(vel_losses_v4.numpy(), axis=0), linewidth=2)
    ax1.plot(time, np.mean(vel_losses_v5.numpy(), axis=0), linewidth=2)
    ax1.plot(time, np.mean(vel_losses_v6.numpy(), axis=0), linewidth=2)
    ax1.set_title("End2End-TCN Mean Velocity Prediction Error over Time")
    ax1.xlabel("Time (ms)")
    ax1.ylabel("Velocity Prediction Error")
    ax1.legend(["End2End-TCN v3", "End2End-TCN v4", "End2End-TCN v5", "End2End-TCN v6"])
    fig1.savefig("E2E_final_vels.png")
    fig1.show()

    time = np.arange(0, 90 * 10, 10)
    fig2, ax2 = plt.subplots()
    ax2.plot(time, np.mean(rate_losses_v3.numpy(), axis=0), linewidth=2)
    ax2.plot(time, np.mean(rate_losses_v4.numpy(), axis=0), linewidth=2)
    ax2.plot(time, np.mean(rate_losses_v5.numpy(), axis=0), linewidth=2)
    ax2.plot(time, np.mean(rate_losses_v6.numpy(), axis=0), linewidth=2)
    ax2.set_title("End2End-TCN Mean Body Rate Prediction Error over Time")
    ax2.xlabel("Time (ms)")
    ax2.ylabel("Body Rate Prediction Error")
    ax2.legend(["End2End-TCN v3", "End2End-TCN v4", "End2End-TCN v5", "End2End-TCN v6"])
    fig2.savefig("E2E_final_rates.png")
    fig2.show()






