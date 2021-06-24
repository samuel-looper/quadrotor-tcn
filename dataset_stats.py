from data_loader import TestSet
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from matplotlib import pyplot as plt

# dataset_stats.py:	Calculate dataset statistics


if __name__ == "__main__":
    # Data initialization
    P = 64
    F = 90
    test_set = TestSet('data/AscTec_Pelican_Flight_Dataset.mat', P, F, full_state=True)

    # Plot test set
    phi = np.asarray(test_set.inputs[:, :, 0]).flatten()
    theta = np.asarray(test_set.inputs[:, :, 1]).flatten()
    psi = np.asarray(test_set.inputs[:, :, 2]).flatten()
    x = np.asarray(test_set.inputs[:, :, 3]).flatten()
    y = np.asarray(test_set.inputs[:, :, 4]).flatten()
    z = np.asarray(test_set.inputs[:, :, 5]).flatten()
    p = np.asarray(test_set.inputs[:, :, 6]).flatten()
    q = np.asarray(test_set.inputs[:, :, 7]).flatten()
    r = np.asarray(test_set.inputs[:, :, 8]).flatten()
    vx = np.asarray(test_set.inputs[:, :, 9]).flatten()
    vy = np.asarray(test_set.inputs[:, :, 10]).flatten()
    vz = np.asarray(test_set.inputs[:, :, 11]).flatten()
    q1 = np.asarray(test_set.inputs[:, :, 12]).flatten()
    q2 = np.asarray(test_set.inputs[:, :, 13]).flatten()
    q3 = np.asarray(test_set.inputs[:, :, 14]).flatten()
    q4 = np.asarray(test_set.inputs[:, :, 15]).flatten()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, 'blue', linewidth=1)
    ax.set_title('Flight Path')
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(0, 4)
    ax.set_xlabel('X-position (m)')
    ax.set_ylabel('Y-position (m)')
    ax.set_zlabel('Z-position (m)')
    plt.show()
    plt.savefig('test_flight_path.png')
    i = 0
    for var in [phi, theta, psi, x, y, z, p, q, r, vx, vy, vz, q1, q2, q3, q4]:
        n, bins, patches = plt.hist(var, 50, density=True, facecolor='g', alpha=0.75)
        plt.title('Variable Distribution')
        plt.savefig("{}".format(i))
        plt.show()
        i+=1