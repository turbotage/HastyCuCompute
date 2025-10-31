import torch

import hastycompute.seq_design.yarnball.yarnball_design as ybd
import hastycompute.utils.torch_utils as torch_utils
import hastycompute.plot.traj_plots as tp

if __name__ == "__main__":
    settings = ybd.YarnballSettings(device=torch.device('cpu'))
    settings.nshot = 3
    grad = ybd.initialize_yarnball_gradient(settings, device=torch.device('cpu'))

    tp.show_trajectory(grad.detach().transpose(1,2).numpy(), 0, figure_size = 8)

    print(grad.shape)
    print(grad)

