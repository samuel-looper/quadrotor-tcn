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
    global_fs_error = np.zeros((50, 1))
    global_vel_error = np.zeros((50, 1))
    global_rate_error = np.zeros((50, 1))
    global_pos_error = np.zeros((50, 1))
    global_ang_error = np.zeros((50, 1))

    # Data initialization
    for batch_size in range(2, 50):
        test_set = SinglePredDatasetTest('data/AscTec_Pelican_Flight_Dataset.mat', batch_size=batch_size)
        n = test_set.inputs.shape[0]
        torques_rec = np.zeros((25, 4))
        state_rec = np.zeros((n, 12))
        full_error_rec = np.zeros((n, 1))
        vel_error_rec = np.zeros((n, 1))
        rate_error_rec = np.zeros((n, 1))
        pos_error_rec = np.zeros((n, 1))
        ang_error_rec = np.zeros((n, 1))
        single_state_rec = np.zeros((batch_size, 12))

        for i in range(n):
            input = test_set.inputs[i, :, :]
            init_condition = np.expand_dims(input[0, :], axis=1)
            final_label = np.expand_dims(test_set.labels[i, batch_size-1, :], axis=1)
            model = WhiteBoxModel(l, d, m, kt, kr, ixx, iyy, izz, init_state=init_condition[:12])

            for j in range(batch_size):
                model.update_thrust(input[j, 12:16])
                model.update_torques()
                model.update(0.01)
                single_state_rec[j, :] = np.vstack((model.ang, model.pos, model.rate, model.vel))[:, 0]

            new_state = np.vstack((model.ang, model.pos, model.rate, model.vel))
            full_error_rec[i] = mse_loss(final_label, new_state)
            vel_error_rec[i] = mse_loss(final_label[9:12], new_state[9:12])
            rate_error_rec[i] = mse_loss(final_label[6:9], new_state[6:9])
            pos_error_rec[i] = mse_loss(final_label[3:6], new_state[3:6])
            ang_error_rec[i] = mse_loss(final_label[:3], new_state[:3])

            if i % 100 == 0:
                print("Interation: {}".format(i))
                print("Full State Error: {} \n".format(full_error_rec[i]))

        print("average error: {}".format(np.mean(full_error_rec)))
        print("velocity error: {}".format(np.mean(vel_error_rec)))
        print("rate error: {}".format(np.mean(rate_error_rec)))
        global_fs_error[batch_size] = np.mean(full_error_rec)
        global_vel_error[batch_size] = np.mean(vel_error_rec)
        global_rate_error[batch_size] = np.mean(rate_error_rec)
        global_pos_error[batch_size] = np.mean(pos_error_rec)
        global_ang_error[batch_size] = np.mean(ang_error_rec)

    plt.plot(global_fs_error)
    plt.title('Full State Error for N-Step Prediction')
    plt.xlabel("# of prediction steps")
    plt.ylabel("Full State Error")
    plt.savefig("full_error.png")
    plt.show()

    plt.plot(global_vel_error)
    plt.title('Velocity Error for N-Step Prediction')
    plt.xlabel("# of prediction steps")
    plt.ylabel("Velocity Error (m/s)")
    plt.savefig("vel_error.png")
    plt.show()

    plt.plot(global_rate_error)
    plt.title('Body Rate Error for N-Step Prediction')
    plt.xlabel("# of prediction steps")
    plt.ylabel("Body Rate Error (rad/s)")
    plt.savefig("rate_error.png")
    plt.show()

    plt.plot(global_pos_error)
    plt.title('Position Error for N-Step Prediction')
    plt.xlabel("# of prediction steps")
    plt.ylabel("Position Error (m)")
    plt.savefig("pos_error.png")
    plt.show()

    plt.plot(global_ang_error)
    plt.title('Euler Angle Error for N-Step Prediction')
    plt.xlabel("# of prediction steps")
    plt.ylabel("Ang Error (rad)")
    plt.savefig("ang_error.png")
    plt.show()

    # Analysis of last test flight
    preds = single_state_rec
    labels = test_set.labels[i, :, :]
    full_error_vec = np.zeros((batch_size, 1))
    ang_error_vec = np.zeros((batch_size, 1))
    pos_error_vec = np.zeros((batch_size, 1))
    rate_error_vec = np.zeros((batch_size, 1))
    vel_error_vec = np.zeros((batch_size, 1))
    t = np.arange(0, batch_size*0.01, 0.01)
    for k in range(batch_size):
        full_error_vec[k] = mse_loss(preds[k, :], labels[k, :])
        ang_error_vec[k] = mse_loss(preds[k, :3], labels[k, :3])
        pos_error_vec[k] = mse_loss(preds[k, 3:6], labels[k, 3:6])
        rate_error_vec[k] = mse_loss(preds[k, 6:9], labels[k, 6:9])
        vel_error_vec[k] = mse_loss(preds[k, 9:], labels[k, 9:])

    plt.plot(t, full_error_vec)
    plt.title('Full State Error over time (25-step)')
    plt.xlabel("Time (s)")
    plt.ylabel("Full State Error")
    plt.savefig("full_error_25.png")
    plt.show()

    plt.plot(t, ang_error_vec)
    plt.title('Angle Error over time (25-step)')
    plt.xlabel("Time (s)")
    plt.ylabel("Angle error (rad)")
    plt.savefig("ang_error_25.png")
    plt.show()

    plt.plot(t, pos_error_vec)
    plt.title('Position Error over time (25-step)')
    plt.xlabel("Time (s)")
    plt.ylabel("Position Error (m)")
    plt.savefig("pos_error_25.png")
    plt.show()

    plt.plot(t, rate_error_vec)
    plt.title('Body Rate Error over time (25-step)')
    plt.xlabel("Time (s)")
    plt.ylabel("Body Rate Error (rad/s)")
    plt.savefig("rate_error_25.png")
    plt.show()

    plt.plot(t, vel_error_vec)
    plt.title('Velocity Error over time (25-step)')
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity Error (m/s)")
    plt.savefig("vel_error_25.png")
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(preds[:, 3], preds[:, 4], preds[:, 5], 'blue', linewidth=1)
    ax.plot3D(labels[:, 3], labels[:, 4], labels[:, 5], 'red', linewidth=1)
    ax.set_title('Flight Path')
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(0, 4)
    ax.set_xlabel('X-position (m)')
    ax.set_ylabel('Y-position (m)')
    ax.set_zlabel('Z-position (m)')
    ax.legend()
    plt.savefig('test_flight_path.png')
    plt.show()
