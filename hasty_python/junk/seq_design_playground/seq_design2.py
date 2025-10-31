import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import math

import pypulseq as pp
import mrinufft as mn

import scipy as sp
from scipy.interpolate import CubicSpline
import scipy.special as sps


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from traj_utils import show_trajectory, show_trajectories



# Trajectory parameters
Nc = 120  # Number of shots
Ns = 500  # Number of samples per shot
in_out = False  # Choose between in-out or center-out trajectories
tilt = "uniform"  # Angular distance between shots
nb_repetitions = 6  # Number of stacks, rotations, cones, shells etc.
nb_revolutions = 1  # Number of revolutions for base trajectories

# Display parameters
figure_size = 15  # Figure size for trajectory plots
subfigure_size = 6  # Figure size for subplots
one_shot = -5  # Highlight one shot in particular


#for shot in range(0, Nc, 10):
#    show_trajectory(trajectory, figure_size=figure_size, one_shot=shot)
#trajectory *= (resolution * delta_k / np.abs(trajectory).max())


def plot31(slew, title=''):
    plt.figure()
    plt.plot(slew[0, :], 'r-*')
    plt.plot(slew[1, :], 'g-*')
    plt.plot(slew[2, :], 'b-*')
    plt.title(title)
    plt.show()

def create_traj_grad_slew(nshots, resolution, fov, system):

    delta_k = 1.0 / fov

    def run_one(ns, nsamp):
        traj = mn.initialize_3D_seiffert_spiral(ns, nsamp, curve_index=0.5, in_out=False)
        traj *= (0.5 * resolution * delta_k / np.abs(traj).max())

        tmax = nsamp * system.grad_raster_time

        CS = CubicSpline(np.linspace(0, tmax, nsamp), traj, axis=1)

        type = 'myhuber'
        oversampling = 4
        if type == 'exp':
            ac = -np.log(1.0 - (0.99**0.25)) / (0.5*tmax)
            tinterp = np.linspace(0, tmax, nsamp*oversampling)
            tinterp = tinterp * ((1.0 - np.exp(-ac*tinterp))**4)
        elif type == 'huber':
            delta = 1e-6
            tinterp = np.linspace(0, tmax, nsamp*oversampling)
            tinterp = sps.pseudo_huber(delta, tinterp)
        elif type == 'myhuber':
            doffset = 0.2
            doffset_inv = 0.95 * (1.0 + doffset)
            tinterp = np.linspace(0, tmax * doffset_inv, int(nsamp * doffset_inv * oversampling))
            delta = doffset * tmax
            coef = 3.0
            tinterp = delta * ((1.0 + (tinterp / delta)**coef)**(1.0/coef) - 1.0)
        elif type == 'none':
            tinterp = np.linspace(0, tmax, nsamp*oversampling)

        plot_tinterp = False
        if plot_tinterp:
            plt.figure()
            plt.plot(np.linspace(0, tmax, nsamp), 'r-')
            plt.plot(tinterp, 'b-')
            plt.show()

        traj = CS(tinterp)[:,::oversampling,:].transpose(0,2,1)
        new_nsamp = traj.shape[-1]
        grad = np.zeros((ns, 3, new_nsamp-1))
        slew = np.zeros((ns, 3, new_nsamp-1))

        for i in range(ns):

            g, s = pp.traj_to_grad(traj[i,...], raster_time=system.grad_raster_time)

            plot_sgt = True
            if plot_sgt:
                plot31(s, 'Slew Rate')
                plot31(g, 'Gradient')
                plot31(traj[i,...], 'Trajectory')

                v = np.sqrt(np.sum(np.square(g), axis=0))

                plt.figure()
                plt.plot(v)
                plt.title('K-space speed')
                plt.show()

            grad[i, ...] = g
            slew[i, ...] = s

        print('TMax: ', tmax * 1000.0, ' ms')

        return traj, grad, slew
    

    traj, grad, slew = run_one(nshots, 1200)
    traj = traj.transpose(0, 2, 1)
    print('Current Slew Rate: ', np.max(np.abs(slew)), 'System Max: ', system.max_slew)
    traj_plot = traj * 0.5 / np.abs(traj).max()
    show_trajectory(traj_plot, figure_size=figure_size, one_shot=one_shot)

    return traj, grad, slew


if True:
    system = pp.Opts(max_grad=80, grad_unit='mT/m', max_slew=200, slew_unit='T/m/s', B0=3.0)
    seq = pp.Sequence(system=system)

    FOV = 220e-3
    delta_k = 1.0 / FOV
    resolution = 256

    ktraj = np.zeros((3, int(3e-3 / system.grad_raster_time)), dtype=np.float32)
    ktraj[0, :] = np.linspace(0, 0.5 * delta_k * resolution, ktraj.shape[1])

    g, s = pp.traj_to_grad(ktraj, raster_time=system.grad_raster_time)
    print('Current Slew Rate: ', np.max(np.abs(s)), 'System Max: ', system.max_slew)




    create_traj_grad_slew(120, resolution, FOV, system)


