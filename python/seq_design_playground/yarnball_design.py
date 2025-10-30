import torch
import torch.nn as nn
import rotation as rot

import gradient_design as gd

from scipy.stats.sampling import TransformedDensityRejection
spdf = rot.SurfacePDF()
rng = TransformedDensityRejection(spdf, center=0.5, domain=(0.0, 1.0))

def get_random_yarnball_angles(nshot):
    spdf = rot.SurfacePDF()
    rng = TransformedDensityRejection(spdf, center=0.5, domain=(0.0, 1.0))

    theta_tilt = torch.pi * torch.tensor(rng.rvs(nshot))
    phi_tilt = 2 * torch.pi * torch.rand(nshot, dtype=torch.float64)
    return (theta_tilt, phi_tilt)

class YarnballSettings:
    def __init__(self):
        self.nshot = 20
        self.nsamp=4000
        self.nb_revs = 17
        self.nb_folds = 5
        self.add_rand_perturb = False
        self.rand_perturb_factor = 0.0
        self.rho_lambda = None
        self.tilts = get_random_yarnball_angles(self.nshot)
        self.angle_tlag_factor = 0.05
        self.rho_tlag_factor = 0.05
        self.omega_tlag_power = 0.8

    def set_nshot(self, nshot: int):
        self.nshot = nshot
        self.tilts = get_random_yarnball_angles(self.nshot)

def initialize_yarnball_gradient(settings: YarnballSettings):
    theta_tilt, phi_tilt = settings.tilts

    t = torch.linspace(0,1,settings.nsamp, dtype=torch.float64)
    lag_length = max(settings.nsamp // 100,1)
    t_lagged =  torch.cat([
                    torch.zeros(lag_length), 
                    (torch.square(t) / (t + settings.angle_tlag_factor))[:-lag_length]
                ])
    

    if settings.rho_lambda is None:
        rho = torch.square(t) / (t + settings.rho_tlag_factor)
    else:
        rho = settings.rho_lambda(t)

    omega = 2*torch.pi*settings.nb_revs
    omega = omega * torch.pow(t_lagged, settings.omega_tlag_power)

    x = rho*torch.cos(omega)
    y = rho*torch.sin(omega)
    z = torch.zeros_like(t)
    grad = torch.stack([x,y,z],dim=0)

    rot_angles = torch.arange(settings.nsamp) * torch.pi * settings.nb_folds / settings.nsamp
    adir = torch.stack([
                    torch.cos(t_lagged), 
                    torch.sin(t_lagged), 
                    torch.zeros_like(t_lagged)
                    ], dim=-1)
    grad = torch.bmm(rot.Ra(adir, rot_angles), grad.transpose(1,0).unsqueeze(-1)).squeeze(-1)

    rmat = torch.bmm(rot.Rz(phi_tilt), rot.Ry(theta_tilt))

    grad = torch.bmm(rmat, grad.transpose(1,0).expand(settings.nshot,-1,-1)).transpose(1,0)

    return grad.transpose(0,1)

class YarballTimeDynamicSegment(gd.GradientTimeDynamicSegment):
    def __init__(self, settings: YarnballSettings, NGRT: int):
        super().__init__(NGRT)
        self.settings = settings

    def output_shape(self):
        return (self.settings.nshot, 3, self.NGRT)

    def forward(self):
        self.settings.nsamp = self.NGRT
        grad = initialize_yarnball_gradient(self.settings)
        return grad
        

if __name__ == "__main__":
    #settings = YarnballSettings()
    grad = initialize_yarnball_gradient(settings)

    if True:
        import os
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        import traj_utils as tu

        tu.show_trajectory(grad.detach().transpose(1,2).numpy(), 0, figure_size = 8)
