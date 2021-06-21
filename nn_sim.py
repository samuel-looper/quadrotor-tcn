from data_loader import TestSet
import numpy as np
from torch.utils.data import DataLoader
import torch
from End2EndNet import E2ESingleStepTCNv4
from scipy import integrate
from matplotlib import pyplot as plt


def plot_flights(pred, label, image_count):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(pred[:, 3], pred[:, 4], pred[:, 5], 'blue', linewidth=1)
    ax.plot3D(label[:, 3], label[:, 4], label[:, 5], 'red', linewidth=1)
    ax.legend(["Predicted Flight Path", "Actual Flight Path"])
    ax.set_title('Flight Path')

    xlower = np.amin(label[:, 3]) - np.abs(np.amin(label[:, 3])) * 0.1
    xupper = np.amax(label[:, 3]) + np.abs(np.amax(label[:, 3])) * 0.1
    ylower = np.amin(label[:, 4]) - np.abs(np.amin(label[:, 4])) * 0.1
    yupper = np.amax(label[:, 4]) + np.abs(np.amax(label[:, 4])) * 0.1
    zlower = np.amin(label[:, 5]) - np.abs(np.amin(label[:, 5])) * 0.1
    zupper = np.amax(label[:, 5]) + np.abs(np.amax(label[:, 5])) * 0.1

    xrange = xupper-xlower
    yrange = yupper-ylower
    zrange = zupper-zlower
    maxrange = max(xrange, yrange, zrange)

    xlower -= (maxrange-xrange)/2
    xupper += (maxrange - xrange) / 2
    ylower -= (maxrange - yrange) / 2
    yupper += (maxrange - yrange) / 2
    zlower -= (maxrange - zrange) / 2
    zupper += (maxrange - zrange) / 2



    ax.set_xlim3d(xlower, xupper)
    ax.set_ylim3d(ylower, yupper)
    ax.set_zlim3d(zlower, zupper)
    ax.set_xlabel('X-position (m)')
    ax.set_ylabel('Y-position (m)')
    ax.set_zlabel('Z-position (m)')
    plt.show()
    fig.savefig('neural_net_sim{}.png'.format(image_count))

class BlackBoxModel:
    def __init__(self, vels, init_state=np.zeros((6, 1))):
        # Initialize variables to store state of white box model
        self.ang = init_state[0:3]              # Angular position (XYZ Euler Angle)
        self.pos = init_state[3:6]              # Linear Position


        self.step = 0
        self.vels = vels

        # Initialize integration method for discrete time dynamics
        self.ode = integrate.ode(self.state_dot).set_integrator('vode', nsteps=500, method='bdf')

    def state_dot(self, time, state):
        rate = self.vels[:3, self.step]
        vel = self.vels[3:, self.step]

        # Calculate rotation matrix from body frame to inertial frame
        s_phi = np.sin(self.ang[0]).item()
        c_phi = np.cos(self.ang[0]).item()
        c_theta = np.cos(self.ang[1]).item()

        # Calculate Euler angle time derivatives
        M = np.asarray([[1, 0, -s_phi], [0, c_phi, s_phi * c_theta], [0, -s_phi, c_theta * c_phi]])
        ang_dot = np.dot(np.linalg.inv(M), rate)

        # Concatenate into final state derivative vector
        state_dot = np.concatenate([ang_dot, vel])
        return state_dot

    def update(self, dt):
        init_state = np.concatenate([self.ang, self.pos])                       # Set initial state for integration
        self.ode.set_initial_value(init_state, 0)                               # Initialize ODE
        updated_state = self.ode.integrate(self.ode.t + dt)                     # Integrate from t to t+dt

        self.ang = updated_state[0:3]
        self.pos = updated_state[3:6]


if __name__ == "__main__":

    lookback = 64
    pred_steps = 90

    test_set = TestSet('data/AscTec_Pelican_Flight_Dataset.mat', lookback, pred_steps, full_set=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=0)

    vel_model = E2ESingleStepTCNv4(lookback, pred_steps)
    vel_model.load_state_dict(torch.load('./E2E_v4.pth', map_location=torch.device("cpu")))
    vel_model.train(False)
    vel_model.eval()
    outliers = 0
    count = 0
    image_count = 0
    outlier_range = np.zeros((16, 1))
    inlier_range = np.zeros((16))

    with torch.no_grad():
        for data in test_loader:
            raw_input = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2)  # Load Input data
            label = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2)  # Load labels

            feedforward = torch.zeros(label.shape)
            feedforward[:, 12:, :] = label[:, 12:, :]
            input = torch.cat((raw_input, feedforward), 2)

            pred_vels = vel_model(input)
            init_vel = torch.zeros((1, 6, 1))
            init_vel[:, :, 0] = raw_input[:, 6:12, -1]
            vels = torch.cat((init_vel, pred_vels), 2)
            black_box_model = BlackBoxModel(vels[0, :, :].numpy(), raw_input[0, :6, -1].numpy())
            state_rec = np.zeros((pred_steps+1, 12))
            state_rec[:, 6:] = black_box_model.vels.T

            for i in range(pred_steps):
                state_rec[i, 0:3] = black_box_model.ang
                state_rec[i, 3:6] = black_box_model.pos
                black_box_model.update(0.01)
                black_box_model.step += 1
            state_rec[pred_steps, 0:3] = black_box_model.ang
            state_rec[pred_steps, 3:6] = black_box_model.pos

            state_label = label[0, :12, :].numpy().T
            state_error = (state_rec[:pred_steps, :] - state_label) ** 2
            full_sequence = np.concatenate((raw_input.numpy(), label.numpy()), axis=2)
            inlier_range += full_sequence[0, :, :].ptp(axis=1)/len(test_set)
            count+=1
            if count % 10 == 0:
                print(count)
            if state_error.sum() > 50:
                ranges = np.expand_dims(full_sequence[0, :, :].ptp(axis=1), axis=1)
                outlier_range = np.concatenate((outlier_range, ranges), axis=1)


            if state_error.sum() < 1:
                print("inliner")
                image_count += 1
                plot_flights(state_rec[:pred_steps, :], state_label, image_count)


    np.savetxt("outlier ranges.csv", outlier_range)
    print(inlier_range)
