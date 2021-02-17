from WhiteBoxModel import WhiteBoxModel
from data_loader import SinglePredDatasetTest
import numpy as np
from matplotlib import pyplot as plt

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

    # Data initialization
    test_set = SinglePredDatasetTest('data/AscTec_Pelican_Flight_Dataset.mat')
    n = test_set.inputs.shape[0]
    model = WhiteBoxModel(l, d, m, kt, kr, ixx, iyy, izz)
    torques_rec = np.zeros((n, 4))
    state_rec = np.zeros((n, 12))
    full_error_rec = np.zeros((n, 1))
    vel_error_rec = np.zeros((n, 1))
    rate_error_rec = np.zeros((n, 1))

    for i in range(n):
        input = np.expand_dims(test_set.inputs[i], axis=1)
        label = np.expand_dims(test_set.labels[i], axis=1)
        model = WhiteBoxModel(l, d, m, kt, kr, ixx, iyy, izz, init_state=input[:12])
        model.update_thrust(input[12:16])
        model.update_torques()
        model.update(0.01)

        new_state = np.vstack((model.ang, model.pos, model.rate, model.vel))
        torques_rec[i, :] = model.torques[:,0]
        state_rec[i, :] = new_state[:,0]
        full_error_rec[i] = mse_loss(label, new_state)
        vel_error_rec[i] = mse_loss(label[9:12], new_state[9:12])
        rate_error_rec[i] = mse_loss(label[6:9], new_state[6:9])

        if i % 100 == 0:
            print("Interation: {}".format(i))
            print("Full State Error: {} \n".format(full_error_rec[i]))

    # data = np.hstack((torques_rec, state_rec, full_error_rec, vel_error_rec,rate_error_rec))
    # np.savetxt("test_set.csv", data)

    fig = plt.figure()

    n, bins, patches = plt.hist(full_error_rec, 50, density=False, facecolor='g', alpha=0.75)
    plt.title('Full State Error Distribution')
    plt.savefig("full_state_error.png")
    plt.show()

    n, bins, patches = plt.hist(vel_error_rec, 50, density=False, facecolor='g', alpha=0.75)
    plt.title('Velocity Error Distribution')
    plt.savefig("velocity_error.png")
    plt.show()

    n, bins, patches = plt.hist(rate_error_rec, 50, density=False, facecolor='g', alpha=0.75)
    plt.title('Body Rates Error Distribution')
    plt.savefig("rates_error.png")
    plt.show()

    n, bins, patches = plt.hist(torques_rec[:, 0], 50, density=False, facecolor='g', alpha=0.75)
    plt.title('Torque Distribution (Motor 1)')
    plt.savefig("t1.png")
    plt.show()

    n, bins, patches = plt.hist(torques_rec[:, 1], 50, density=False, facecolor='g', alpha=0.75)
    plt.title('Torque Distribution (Motor 2)')
    plt.savefig("t2.png")
    plt.show()

    n, bins, patches = plt.hist(torques_rec[:, 2], 50, density=False, facecolor='g', alpha=0.75)
    plt.title('Torque Distribution (Motor 3)')
    plt.savefig("t3.png")
    plt.show()

    n, bins, patches = plt.hist(torques_rec[:, 3], 50, density=False, facecolor='g', alpha=0.75)
    plt.title('Torque Distribution (Motor 4)')
    plt.savefig("t4.png")
    plt.show()



