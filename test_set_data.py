from data_loader import SinglePredDatasetTest
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from numpy import savetxt

if __name__ == "__main__":
    # Data initialization
    test_set = SinglePredDatasetTest('data/AscTec_Pelican_Flight_Dataset.mat')

    # Plot test set
    phi = test_set.inputs[:, 0]
    theta = test_set.inputs[:, 1]
    psi = test_set.inputs[:, 2]
    x = test_set.inputs[:, 3]
    y = test_set.inputs[:, 4]
    z= test_set.inputs[:, 5]
    p = test_set.inputs[:, 6]
    q = test_set.inputs[:, 7]
    r = test_set.inputs[:, 8]
    vx = test_set.inputs[:, 9]
    vy = test_set.inputs[:, 10]
    vz = test_set.inputs[:, 11]
    q1 = test_set.inputs[:, 12]
    q2 = test_set.inputs[:, 13]
    q3 = test_set.inputs[:, 14]
    q4 = test_set.inputs[:, 15]

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

    np.savetxt("test_set.csv", test_set.inputs)
