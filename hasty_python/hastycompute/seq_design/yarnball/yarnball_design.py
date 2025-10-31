import torch
import torch.nn as nn

import seq_design.gradient_design as gd
import utils.rotation as rot

from scipy.stats.sampling import TransformedDensityRejection
spdf = rot.SurfacePDF()
rng = TransformedDensityRejection(spdf, center=0.5, domain=(0.0, 1.0))

def get_random_yarnball_angles(nshot, device=torch.device('cpu')):
    spdf = rot.SurfacePDF()
    rng = TransformedDensityRejection(spdf, center=0.5, domain=(0.0, 1.0))

    theta_tilt = torch.pi * torch.tensor(rng.rvs(nshot)).to(device).to(dtype=torch.float64)
    phi_tilt = 2 * torch.pi * torch.rand(nshot, dtype=torch.float64, device=device)
    return (theta_tilt, phi_tilt)

class YarnballSettings:
    def __init__(self, device=torch.device('cpu')):
        self.nshot = 20
        self.nsamp=4000
        self.nb_revs = 17
        self.nb_folds = 5
        self.add_rand_perturb = False
        self.rand_perturb_factor = 0.0
        self.rho_lambda = None
        self.tilts = get_random_yarnball_angles(self.nshot, device=device)
        self.angle_tlag_factor = 0.05
        self.rho_tlag_factor = 0.05
        self.omega_tlag_power = 0.8
        self.device=device

    def set_nshot(self, nshot: int):
        self.nshot = nshot
        self.tilts = get_random_yarnball_angles(self.nshot, device=self.device)

def initialize_yarnball_gradient(settings: YarnballSettings, device=torch.device('cpu')):
    theta_tilt, phi_tilt = settings.tilts

    t = torch.linspace(0,1,settings.nsamp, dtype=torch.float64, device=device)
    lag_length = max(settings.nsamp // 100,1)
    t_lagged =  torch.cat([
                    torch.zeros(lag_length, dtype=torch.float64, device=device), 
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

    rot_angles = torch.arange(settings.nsamp, device=device) * torch.pi * settings.nb_folds / settings.nsamp
    adir = torch.stack([
                    torch.cos(t_lagged), 
                    torch.sin(t_lagged), 
                    torch.zeros_like(t_lagged)
                    ], dim=-1)
    grad = torch.bmm(rot.Ra(adir, rot_angles), grad.transpose(1,0).unsqueeze(-1)).squeeze(-1)

    rmat = torch.bmm(rot.Rz(phi_tilt), rot.Ry(theta_tilt))

    grad = torch.bmm(rmat, grad.transpose(1,0).expand(settings.nshot,-1,-1)).transpose(1,0)

    return grad.transpose(0,1)

class YarnballTimeDynamicSegment(gd.GradientTimeDynamicSegment):
    def __init__(self, settings: YarnballSettings, NGRT: int, device=torch.device('cpu')):
        super().__init__(NGRT)
        self.settings = settings
        self.device = device

    def output_shape(self):
        return (self.settings.nshot, 3, self.NGRT)

    def forward(self):
        self.settings.nsamp = self.NGRT
        grad = initialize_yarnball_gradient(self.settings, device=self.device)
        return grad
        

if __name__ == "__main__":

    if True:
        settings = YarnballSettings()
        grad = initialize_yarnball_gradient(settings)

        import hasty_python.plot.traj_plots as tp

        tp.show_trajectory(grad.detach().transpose(1,2).numpy(), 0, figure_size = 8)
