import numpy as np

import torch
from model import ConvSkip


class TrimWrapper:
    def __init__(self):
        self.trim_est = 0

        self.k = 27
        self.b = 0.108
        self.R = 0.0318
        self.dt = 1/15.

        self.model = self.load_model()

    def undistort_action(self, u_l, u_r):
        u_l = u_l*(1 - self.trim_est)
        u_r = u_r*(1 + self.trim_est)
        return u_l, u_r

    def estimate_trim(self, x):
        # x = [delta_phi, u_l, u_r]
        # Remove data when DB accelerates
        x = x[30:]

        # Remove data when there is no delta angle
        mask = x[:, 0] != 0
        x = x[mask]

        # Consider ideal kinematics
        omega_l = x[:, 1] * self.k
        omega_r = x[:, 2] * self.k
        omega = (omega_r - omega_l) * self.R / self.b
        v = self.R / 2 * (omega_l + omega_r)

        # Compute an estimate for the real omega
        omega_meas = x[:, 0] / self.dt

        # Computing the trim paramter
        t = np.divide(omega_meas - omega, v) * self.b / 2.
        t = np.expand_dims(t, 1)

        # Discard outliers
        mask = np.abs(t) < 1
        t = t[mask]

        # Compute average over samples
        t_est = np.mean(t)

        return t_est

    def get_delta_phi(self, img0,  img1):
        img0 = torch.from_numpy(img0)[None, None, :, :]
        img1 = torch.from_numpy(img1)[None, None, :, :]

        img = torch.cat((img0, img1), 1).double()/255.
        print(img)

        delta_phi = self.model(img)
        return delta_phi

    def load_model(self):
        checkpoint = torch.load('/workspace/ConvSkip_tmp.pth', map_location=torch.device('cpu'))
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        return model.eval()
