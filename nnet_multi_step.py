import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from End2EndNet import E2ESingleStepTCNv4
from FullHybridNet import QuadrotorDynamicsFH
from MotorHybridNet import QuadrotorDynamicsMH
from AccelErrorNet import QuadrotorDynamicsAE
from torchdiffeq import odeint
from data_loader import TestSet


def recurrent_test(test_loader, model, rate_losses, vel_losses, name):
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
            print("Sample #{}".format(i))

        np.savetxt("{}_test_error_rates.csv".format(name), rate_losses.numpy())
        np.savetxt("{}_test_results_vels.csv".format(name), vel_losses.numpy())


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    l = 0.211  # length (m)
    d = 1.7e-5  # blade parameter
    m = 1  # mass (kg)
    kt = 2.35e-14  # translational drag coefficient
    kr = 0.0099  # rotational drag coefficient
    ixx = 0.002  # moment of inertia about X-axis
    iyy = 0.002  # moment of inertia about Y-axis
    izz = 0.001  # moment of inertia about Z-axis
    lookback = 64
    pred_steps = 90
    bs = 1
    # Data initialization
    test_set = TestSet('data/AscTec_Pelican_Flight_Dataset.mat', lookback, pred_steps, full_set=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=bs, shuffle=True, num_workers=0)
    n = int(len(test_set) / bs)
    print("Testing Length: {}".format(n))

    name = "E2E_v4"
    model = E2ESingleStepTCNv4(lookback, pred_steps)
    model.load_state_dict(torch.load('./{}.pth'.format(name)))
    model.train(False)
    model.eval()
    vel_losses_vE2E = torch.zeros((n, pred_steps))
    rate_losses_vE2E = torch.zeros((n, pred_steps))
    conv_test(test_loader, model, rate_losses_vE2E, vel_losses_vE2E, name)

    name = "Combined-Hybrid"
    model = QuadrotorDynamicsFH(l, m, d, kt, kr, ixx, iyy, izz, lookback, 1)
    model.load_state_dict(torch.load('./{}.pth'.format(name)))
    model.train(False)
    model.eval()
    vel_losses_vCH = torch.zeros((n, pred_steps))
    rate_losses_vCH = torch.zeros((n, pred_steps))
    recurrent_test(test_loader, model, rate_losses_vCH, vel_losses_vCH, name)

    name = "Motor-Hybrid"
    model = QuadrotorDynamicsMH(l, m, d, kt, kr, ixx, iyy, izz, lookback, 1)
    model.load_state_dict(torch.load('./{}.pth'.format(name)))
    model.train(False)
    model.eval()
    vel_losses_vMH = torch.zeros((n, pred_steps))
    rate_losses_vMH = torch.zeros((n, pred_steps))
    recurrent_test(test_loader, model, rate_losses_vMH, vel_losses_vMH, name)

    name = "AccelError-Hybrid"
    model = QuadrotorDynamicsAE(l, m, d, kt, kr, ixx, iyy, izz, lookback, 1)
    model.load_state_dict(torch.load('./{}.pth'.format(name)))
    model.train(False)
    model.eval()
    vel_losses_vAE = torch.zeros((n, pred_steps))
    rate_losses_vAE = torch.zeros((n, pred_steps))
    recurrent_test(test_loader, model, rate_losses_vAE, vel_losses_vAE, name)

    lstm_hybrid_vels = np.genfromtxt('lstm_hybrid_vels.csv', delimiter=',')
    lstm_hybrid_rates = np.genfromtxt('lstm_hybrid_rates.csv', delimiter=',')

    time = np.arange(0, 90*10, 10)
    fig1, ax1 = plt.subplots()
    ax1.plot(time, lstm_hybrid_vels, "o", markersize=2)
    ax1.plot(time, np.mean(vel_losses_vE2E.numpy(), axis=0), "o", markersize=2)
    ax1.plot(time, np.mean(vel_losses_vCH.numpy(), axis=0), "o", markersize=2)
    ax1.plot(time, np.mean(vel_losses_vMH.numpy(), axis=0), "o", markersize=2)
    ax1.plot(time, np.mean(vel_losses_vAE.numpy(), axis=0), "o", markersize=2)
    ax1.set_title("Mean Velocity Prediction Error over Time Model Comparison")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Velocity Prediction Error")
    ax1.legend(["LSTM Hybrid", "End2End-TCN", "Combined-Hybrid-TCN", "Motor-Hybrid-TCN", "AccelError-Hybrid-TCN" ])
    fig1.savefig("overall_final_vels.png")
    # fig1.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(time, lstm_hybrid_rates, "o", markersize=2)
    ax2.plot(time, np.mean(rate_losses_vE2E.numpy(), axis=0), "o", markersize=2)
    ax2.plot(time, np.mean(rate_losses_vCH.numpy(), axis=0), "o", markersize=2)
    ax2.plot(time, np.mean(rate_losses_vMH.numpy(), axis=0), "o", markersize=2)
    ax2.plot(time, np.mean(rate_losses_vAE.numpy(), axis=0), "o", markersize=2)
    ax2.set_title("Mean Body Rate Prediction Error over Time Model Comparison")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Body Rate Prediction Error")
    ax2.legend(["LSTM Hybrid", "End2End-TCN", "Combined-Hybrid-TCN", "Motor-Hybrid-TCN", "AccelError-Hybrid-TCN" ])
    fig2.savefig("overall_final_rates.png")
    # fig2.show()

    # fig3, ax3 = plt.subplots(figsize=(24.0, 6.0))
    # bplot = ax3.boxplot(rate_losses_v4.T, sym="", medianprops=dict(linewidth=3, color='red'), patch_artist=True)
    # ax3.set_title("End2End-TCN v4 Body Rate Prediction Range Over Time")
    # ax3.set_xlabel("Sample #")
    # ax3.set_ylabel("Body Rate Prediction Error")
    # ax3.set_ylim([0, 0.1])
    # for patch in bplot["boxes"]:
    #     patch.set_facecolor("lightblue")
    #
    # fig3.savefig("E2E_v4_rate_range.png")
    # fig3.show()
    #
    # fig4, ax4 = plt.subplots(figsize=(24.0, 6.0))
    # bplot = ax4.boxplot(vel_losses_v4.T, sym="", medianprops=dict(linewidth=3, color='red'), patch_artist=True)
    # ax4.set_title("End2End-TCN v4 Velocity Prediction Range Over Time")
    # ax4.set_xlabel("Sample #")
    # ax4.set_ylabel("Velocity Prediction Error")
    # ax4.set_ylim([0, 0.1])
    # for patch in bplot["boxes"]:
    #     patch.set_facecolor("lightblue")
    #
    # fig4.savefig("E2E_v4_vel_range.png")
    # fig4.show()
    #
    # fig5, ax5 = plt.subplots(figsize=(24.0, 6.0))
    # bplot = ax5.boxplot(rate_losses_v6.T, sym="", medianprops=dict(linewidth=3, color='red'), patch_artist=True)
    # ax5.set_title("End2End-TCN v6 Body Rate Prediction Range Over Time")
    # ax5.set_xlabel("Sample #")
    # ax5.set_ylabel("Body Rate Prediction Error")
    # ax5.set_ylim([0, 0.1])
    # for patch in bplot["boxes"]:
    #     patch.set_facecolor("lightblue")
    #
    # fig5.savefig("E2E_v6_rate_range.png")
    # fig5.show()
    #
    # fig6, ax6 = plt.subplots(figsize=(24.0, 6.0))
    # bplot = ax6.boxplot(vel_losses_v6.T, sym="", medianprops=dict(linewidth=3, color='red'), patch_artist=True)
    # ax6.set_title("End2End-TCN v6 Velocity Prediction Range Over Time")
    # ax6.set_xlabel("Sample #")
    # ax6.set_ylabel("Velocity Prediction Error")
    # ax6.set_ylim([0, 0.1])
    # for patch in bplot["boxes"]:
    #     patch.set_facecolor("lightblue")
    #
    # fig6.savefig("E2E_v6_vel_range.png")
    # fig6.show()




