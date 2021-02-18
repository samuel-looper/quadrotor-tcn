import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from data_loader import SinglePredDatasetTrain
from torch.utils.data import DataLoader
import gc
PATH = './sys_id.pth'


class SysID(nn.Module):
    # optimize specific parameters within the quadrotor dynamics model
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.FloatTensor([[0.0011], [-0.0005], [0.001], [-0.0001]]))
        self.b = torch.nn.Parameter(torch.FloatTensor([[-0.0069], [-0.0069], [-0.0088], [-0.0121]]))
        self.c = torch.nn.Parameter(torch.FloatTensor([[2.2929], [2.5556], [2.2989], [2.5572]]))

        self.a.requires_grad = True
        self.b.requires_grad = True
        self.c.requires_grad = True

    def forward(self, commands):
        n1 = torch.mul(torch.mul(commands, commands), self.a)
        n2 = torch.mul(commands, self.b)
        return n1+n2+self.c


class QuadrotorDynamics(nn.Module):
    def __init__(self, l, m, d, kt, kr, ixx, iyy, izz):
        super().__init__()
        self.l = l
        self.m = m
        self.d = d
        self.kt = kt
        self.kr = kr
        self.I = torch.tensor([[ixx, 0, 0], [0, iyy, 0], [0, 0, izz]])
        self.param_net = SysID()
        self.torque_mat = torch.tensor([[1, 1, 1, 1],
                          [0.707 * self.l, -0.707 * self.l, -0.707 * self.l, 0.707 * self.l],
                          [-0.707 * self.l, -0.707 * self.l, 0.707 * self.l, 0.707 * self.l],
                          [-self.d, self.d, -self.d, self.d]])
        self.select = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [1/self.m, 0, 0, 0]]).type(torch.FloatTensor)
        self.g = torch.tensor([[0], [0], [9.8067]])
        self.iter_count = 0

    def forward(self, t, input):
        # self.iter_count += 1

        commands = input[-4:, :]
        state = input[:-4, :]

        # update thrusts
        thrusts = self.param_net(commands)

        # update torques
        torques = torch.mm(self.torque_mat, thrusts)

        ang = state[0:3, 0]
        rate = state[6:9, :]
        vel = state[9:12, :]
        # Calculate rotation matrix
        s_phi = (torch.sin(ang[0])).item()
        c_phi = (torch.cos(ang[0])).item()
        s_theta = (torch.sin(ang[1])).item()
        c_theta = (torch.cos(ang[1])).item()
        s_psi = (torch.sin(ang[2])).item()
        c_psi = (torch.cos(ang[2])).item()

        rbi = torch.tensor(
            [[c_theta * c_psi, c_psi * s_theta * s_phi - c_phi * s_psi, c_phi * c_psi * s_theta + s_phi * s_psi],
             [c_theta * s_psi, s_psi * s_theta * s_phi + c_phi * c_psi, c_phi * s_psi * s_theta - s_phi * c_psi],
             [-s_theta, c_theta * s_phi, c_theta * c_phi]])

        # gravity_comp = torch.mm(torch.inverse(rbi), self.m * self.g)
        # grav_torque = torch.zeros(torques.shape)
        # grav_torque[0] = gravity_comp[2]
        # Calculate M matrix
        M = torch.tensor([[1, 0, -s_phi], [0, c_phi, s_phi * c_theta], [0, -s_phi, c_theta * c_phi]])

        vel_dot = torch.mm(rbi, torch.mm(self.select, torques)) - self.kt * vel - self.g
        m_inv = torch.inverse(M)
        ang_dot = torch.mm(m_inv, rate)
        rate_dot = torch.mm(torch.inverse(self.I), torques[1:] - torch.cross(rate, torch.mm(self.I, rate), dim=0)
                             - self.kr * rate)

        state_dot = torch.cat([ang_dot, vel, rate_dot, vel_dot, torch.zeros((4, 1))])
        # print(torch.norm(m_inv))
        # if self.iter_count == 50:
        #     print("Imminent Explosion")

        return state_dot


if __name__ == "__main__":
    l = 0.211                               # length (m)
    d = 1.7e-5                              # blade parameter
    m = 1                                   # mass (kg)
    kt = 2.35e-14                           # translational drag coefficient
    kr = 0.0099                             # rotational drag coefficient
    ixx = 0.002                             # moment of inertia about X-axis
    iyy = 0.002                             # moment of inertia about Y-axis
    izz = 0.001                             # moment of inertia about Z-axis

    bs = 16
    lookback = 1
    lr = 0.001
    wd = 0.0005
    epochs = 10
    pred_steps = 1

    tv_set = SinglePredDatasetTrain('data/AscTec_Pelican_Flight_Dataset.mat', lookback, pred_steps, full_set=True)
    train_len = int(len(tv_set) * 0.8)
    val_len = len(tv_set) - train_len
    train_set, val_set = torch.utils.data.random_split(tv_set, [train_len, val_len], torch.Generator())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=0)
    print("Data Loaded Successfully")

    func = QuadrotorDynamics(l, m, d, kt, kr, ixx, iyy, izz)
    func.load_state_dict(torch.load(PATH)) # Load model

    params = list(func.parameters())
    optimizer = optim.Adam(params, lr=lr)
    loss_f = nn.MSELoss()
    train_loss = []
    val_loss = []
    best_loss = 0
    best_epoch = 0
    print("Training Length: {}".format(int(train_len / bs)))

    for epoch in range(1, epochs + 1):
        # Training
        # print("Training")
        func.train(True)
        epoch_train_losses = []
        epoch_train_loss = 0
        epoch_val_losses = []
        i = 0

        for data in train_loader:
            batch_inputs = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2)  # Load Input data
            batch_labels = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2)[:, 6:12, :]  # Load labels
            optimizer.zero_grad()
            pred = torch.zeros(batch_labels.shape)
            for j in range(batch_inputs.shape[0]):
                input = batch_inputs[j, :, :]
                label = batch_labels[j, :, :]
                output = odeint(func, input, torch.tensor([0, 0.01]))
                # print(func.iter_count)
                # func.iter_count = 0
                pred[j, :, :] = output[1, 6:12, :]

            loss = loss_f(pred, batch_labels)
            epoch_train_loss += bs*loss.detach().item() / train_len
            if loss.requires_grad:
                loss.backward()

                optimizer.step()
            i += 1

            if i % 20 == 0:
                print("Training {}% finished".format(round(100*i*bs/train_len, 4)))
                print(epoch_train_loss*train_len/(i*bs))
                # break
            gc.collect()

        train_loss.append(epoch_train_loss)
        print("Training Error for this Epoch: {}".format(epoch_train_loss))

        # Validation
        print("Validation")
        func.train(False)
        func.eval()
        epoch_val_loss = 0
        i = 0
        with torch.no_grad():
            for data in val_loader:
                batch_inputs = torch.transpose(data["input"].type(torch.FloatTensor), 1, 2)  # Load Input data
                batch_labels = torch.transpose(data["label"].type(torch.FloatTensor), 1, 2)[:, 6:12, :]  # Load labels
                optimizer.zero_grad()
                pred = torch.zeros(batch_labels.shape)
                for j in range(batch_inputs.shape[0]):
                    input = batch_inputs[j, :, :]
                    label = batch_labels[j, :, :]
                    output = odeint(func, input, torch.tensor([0, 0.01]))  # This is where the explosion happens
                    # print(func.iter_count)
                    # func.iter_count = 0
                    pred[j, :, :] = output[1, 6:12, :]

                loss = loss_f(pred, batch_labels)
                epoch_val_losses.append(loss.detach().item() )

                i += 1
                if i % 80 == 0:
                    print(i)

            val_loss.append(np.mean(epoch_val_losses))
            print("Validation Error for this Epoch: {}".format(np.mean(epoch_val_losses)))
            if best_epoch == 0 or np.mean(epoch_val_losses) < best_loss:
                best_loss = np.mean(epoch_val_losses)
                best_epoch = epoch
                torch.save(func.state_dict(), PATH)

            # Plotting
            plt.plot(train_loss, linewidth=2)
            plt.plot(val_loss, linewidth=2)
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.legend(["Training Loss", "Validation Loss"])
            plt.savefig("MHybrid_1Step_losses_intermediate.png")
            plt.show()

    print("Training Complete")
    print("Best Validation Error ({}) at epoch {}".format(best_loss, best_epoch))

    # Plot Final Training Errors
    plt.plot(train_loss, linewidth=2)
    plt.plot(val_loss, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend(["Training Loss", "Validation Loss"])
    plt.savefig("MHybrid_1Step_losses.png")
    plt.show()


