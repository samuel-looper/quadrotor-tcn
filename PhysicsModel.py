import numpy as np
from scipy import integrate

# PhysicsModel.py:	Build and simulate physics-based quadrotor models


class PhysicsModel:
    # Physics-based Quadrotor Model
    def __init__(self, l, d, m, kt, kr, ixx, iyy, izz, init_state=np.zeros((12, 1))):
        # Initialize variables to store quadrotor
        self.ang = init_state[0:3]              # Angular position (XYZ Euler Angle)
        self.pos = init_state[3:6]              # Linear Position
        self.rate = init_state[6:9]             # Body-frame rates (Euler Angle Angular Velocity)
        self.vel = init_state[9:12]             # Linear Velocity
        self.torques = np.zeros((4, 1))         # Initialize body torque
        self.thrusts = np.zeros((4, 1))         # Initialize motor thrust

        # Initialize constant values and matrices
        self.kt = kt                                                    # Translational drag coefficient
        self.kr = kr                                                    # Rotational drag coefficient
        self.I = np.asarray([[ixx, 0, 0], [0, iyy, 0], [0, 0, izz]])    # Rotational Inertia Matrix
        self.torque_mat = np.asarray([[1, 1, 1, 1],                     # Vectorized torque calculation matrix
                                        [0.707 * l, -0.707 * l, -0.707 * l, 0.707 * l],
                                        [-0.707 * l, -0.707 * l, 0.707 * l, 0.707 * l],
                                        [-d, d, -d, d]])
        self.select = np.asarray([[0, 0, 0, 0],                         # Vectorized acceleration calculation matrix
                                  [0, 0, 0, 0],
                                  [1 / m, 0, 0, 0]])
        self.g = np.asarray([[0], [0], [9.8067]])                       # Gravitational acceleration vector
        self.thrust_mat = np.asarray([[0.0011, -0.0069, 2.2929],  # Vectorized thrust calculation coef. matrix
                                     [-0.0005, -0.0088, 2.5556],
                                     [0.001, -0.0121, 2.2989],
                                     [-0.0001, -0.0116, 2.5572]])

        # Initialize integration method for discrete time dynamics
        self.ode = integrate.ode(self.state_dot).set_integrator('vode', nsteps=500, method='bdf')

    def update_torques(self):
        # Calculates torques and updates internal variables
        self.torques = np.dot(self.torque_mat, self.thrusts)
        return self.torques

    def wrap_angle(self, ang):
        # To constrain angles from - 2pi to 2pi
        return (np.pi + ang) % (2 * np.pi) - np.pi

    def state_dot(self, time, state):
        # Calculates continuous time derivatives of each of the quadrotor states

        # Calculate rotation matrix from body frame to inertial frame
        s_phi = np.asscalar(np.sin(self.ang[0]))
        c_phi = np.asscalar(np.cos(self.ang[0]))
        s_theta = np.asscalar(np.sin(self.ang[1]))
        c_theta = np.asscalar(np.cos(self.ang[1]))
        s_psi = np.asscalar(np.sin(self.ang[2]))
        c_psi = np.asscalar(np.cos(self.ang[2]))

        rbi = np.asarray(
            [[c_theta * c_psi, c_psi * s_theta * s_phi - c_phi * s_psi, c_phi * c_psi * s_theta + s_phi * s_psi],
             [c_theta * s_psi, s_psi * s_theta * s_phi + c_phi * c_psi, c_phi * s_psi * s_theta - s_phi * c_psi],
             [-s_theta, c_theta * s_phi, c_theta * c_phi]])

        # Calculate Euler angle time derivatives
        M = np.asarray([[1, 0, -s_phi], [0, c_phi, s_phi * c_theta], [0, -s_phi, c_theta * c_phi]])
        ang_dot = np.dot(np.linalg.inv(M), self.rate)

        # Linear Acceleration
        vel_dot = np.dot(rbi, np.dot(self.select, self.torques)) - self.kt * self.vel - self.g

        # Rotational Acceleration
        rate_dot = np.dot(np.linalg.inv(self.I), self.torques[1:] -
                          np.cross(self.rate, np.dot(self.I, self.rate), axis=0) - self.kr * self.rate)

        # Concatenate into final state derivative vector
        state_dot = np.concatenate([ang_dot, self.vel, rate_dot, vel_dot])
        return state_dot

    def update(self, dt):
        init_state = np.concatenate([self.ang, self.pos, self.rate, self.vel])  # Set initial state for integration
        self.ode.set_initial_value(init_state, 0)                               # Initialize ODE
        updated_state = self.ode.integrate(self.ode.t + dt)                     # Integrate from t to t+dt
        self.ang = self.wrap_angle(updated_state[0:3])                          # Wrap angles
        self.ang = updated_state[0:3]
        self.pos = updated_state[3:6]                                          # Update state
        self.rate = updated_state[6:9]
        self.vel = updated_state[9:12]

    def update_thrust(self, motor_cmd):
        # Update thrust values based on motor commands and coefficients from system identification
        cmd_mat = np.vstack([np.square(motor_cmd.T), motor_cmd.T, np.ones((1, 4))])
        self.thrusts = np.expand_dims(np.diagonal(np.dot(self.thrust_mat,cmd_mat)), axis=1)
        return self.thrusts

