"""Functions to initialize 3D trajectories."""

import math
from functools import partial
from typing import Literal
import scipy as sp

import numpy as np
import numpy.linalg as nl
from numpy.typing import NDArray
from scipy.special import ellipj, ellipk
from scipy.interpolate import PchipInterpolator

from mrinufft.trajectories.maths import (
	CIRCLE_PACKING_DENSITY,
	EIGENVECTOR_2D_FIBONACCI,
	R2D,
	Ra,
	Rx,
	Ry,
	Rz,
	generate_fibonacci_circle
)

from mrinufft.trajectories.tools import conify, duplicate_along_axes, epify, precess, stack
from mrinufft.trajectories.trajectory2D import initialize_2D_spiral, initialize_2D_radial
from scipy.stats.sampling import TransformedDensityRejection

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import traj_utils as tu

import matplotlib.pyplot as plt

class SurfacePDF:
	def pdf(self, x: float) -> float:
		return math.sin(math.pi*x)
		# if x <= 0:
		# 	return 0
		# elif x >= 1:
		# 	return 0
		# else:
		# 	return 2*math.sin(math.pi*x)
		
	def dpdf(self, x: float) -> float:
		# if x <= 0:
		# 	return 0
		# elif x >= 1:
		# 	return 0
		# else:
		# 	return math.pi*math.cos(math.pi*x)
		return math.pi*math.cos(math.pi*x)
	
	def mean(self) -> float:
		return 0.5

def initialize_my_yarn_ball(
	Nc: int,
	Ns: int,
	tilt: str | float = "golden",
	nb_revs: float = 5,
	nb_folds: float = 5,
	rho_lambda = None,
	plot=False
	) -> NDArray:


	spdf = SurfacePDF()
	#urng = np.random.default_rng()
	rng = TransformedDensityRejection(spdf, center=0.5, domain=(0.0, 1.0))

	#Nc = 50000
	theta_tilt = np.pi * rng.rvs(Nc)
	phi_tilt = np.random.uniform(0, 2*np.pi, Nc)

	#theta_tilt *= 0.0
	#phi_tilt *= 0.0

	if plot:
		plt.figure()
		plt.hist(theta_tilt, bins=100)
		plt.show()

		plt.figure()
		plt.hist(phi_tilt, bins=100)
		plt.show()

	t = np.linspace(0,1-1e-7,Ns)
	lag_length = Ns//100
	t_lagged = np.concatenate([np.zeros((lag_length,)), (np.square(t) / (t + 0.1))[:-lag_length]])

	omega = 2*np.pi*nb_revs

	if rho_lambda is None:
		rho_0 = 1*(np.square(t) / (t+0.1))
		rho_wiggle = 1#(1 + 0.2*np.sin(2*np.pi*t))
		rho = rho_0*rho_wiggle
	else:
		rho = rho_lambda(t)

	if plot:
		plt.figure()
		plt.plot(t, rho)
		#plt.plot(rho)
		plt.title("Rho profile")
		plt.show()

	omega = omega * (t_lagged)**0.8

	x = rho*np.cos(omega)
	y = rho*np.sin(omega)
	z = np.zeros_like(t)
	traj = np.stack([x,y,z],axis=0)

	rot_angle = np.pi*nb_folds/Ns
	for i in range(1,Ns):
		ti = t_lagged[i]
		rmat = Ra(np.array([np.cos(ti), np.sin(ti), 0.0]), rot_angle*i)
		traj[:,i] = (rmat @ traj[:,i][:,None])[:,0]

	traj = np.concatenate([np.zeros((3,1)), traj[:,:-1]], axis=1)
	traj = traj[None,...]

	for i in range(0,Nc):
		theta = theta_tilt[i]
		phi = phi_tilt[i]
		rmat = Rz(phi) @ Ry(theta)
		traj = np.concatenate([traj, (rmat @ traj[0])[None,...]], axis=0)

	traj = traj[1:].transpose(0,2,1)
	traj = 0.5*traj / np.abs(traj).max()

	if plot:
		tu.show_trajectory(traj, 0, figure_size = 8)

	return traj

def my_yarn_ball_default_rho(a, c, n, m):
	def g(t):
		return np.square(t) * (1 - np.power(t, n)) / (t+a)

	def f(t):
		return np.e * np.exp(-1/np.power(1-np.power(t,m), c))

	def rho(t):
		return g(t) * f(t)

	def rhodt(t):
		h = 1e-6
		return (rho(t+h) - rho(t)) / h

	maximum = sp.optimize.brentq(rhodt, 0.5, 0.96)
	maximum = rho(maximum)

	return lambda t: rho(t) / maximum

def my_yarn_ball_default_rho_2(a, n, b):
	def g(t):
		return np.square(t) * (1 - np.power(t, n)) / (t+a)

	def f(t):
		s = (t - (1-b)) / b
		m = np.minimum(np.maximum(1-s, 0.0), 1.0)
		return np.square(m) * (3 - 2*m)

	def rho(t):
		return g(t) * f(t)

	def rhodt(t):
		h = 1e-6
		return (rho(t+h) - rho(t)) / h

	maximum = sp.optimize.brentq(rhodt, 0.5, 0.98)
	maximum = rho(maximum)

	return lambda t: rho(t) / maximum

def my_yarn_ball_default_rho_3(a, b, c, n, m):
	def g(t):
		return (a - np.power(t - b, 2.0))*(1-np.power(t,n))

	def f(t):
		if isinstance(t, float):
			if t <= 1e-6 or t >= (1.0 - 1e-6):
				return 0.0
			else:
				s = (t - 0.5) / 0.5
				return np.e * np.exp(-1/np.power(1-np.power(s,m), c))
		elif isinstance(t, np.ndarray):
			ret = np.empty_like(t)
			ret[(t <= 1e-6) | (t >= 1.0 - 1e-6)] = 0.0
			geq = (t > 1e-6) & (t < (1.0 - 1e-6))
			s = (t[geq] - 0.5) / 0.5
			ret[geq] = np.e * np.exp(-1/np.power(1-np.power(s,m), c))
			return ret
		else:
			raise ValueError("Input must be float or ndarray")

	def rho(t):
		return g(t) * f(t)

	def rhodt(t):
		h = 1e-6
		return (rho(t+h) - rho(t)) / h

	maximum = sp.optimize.brentq(rhodt, 0.5, 0.98)
	maximum = rho(maximum)

	return lambda t: rho(t) / maximum

def my_yarn_ball_default_rho_4():
	pchip = PchipInterpolator(
		x=[0.0, 0.001, 0.10, 0.3, 0.85, 0.9, 0.95, 0.999,  1.0],
		y=[0.0, 0.0,   0.6,  0.8, 1.0,  0.8, 0.5,  0.0,    0.0]
	)

	return lambda t: pchip(t)
	

if __name__ == "__main__":

	#initialize_my_yarn_ball(Nc=1, Ns=5000, nb_revs=6, nb_folds=3, rho_lambda=my_yarn_ball_default_rho(0.05, 20, 12), plot=True)

	#rholam = my_yarn_ball_default_rho_3(0.8, 0.8, 0.75, 25, 12)
	#rholam = my_yarn_ball_default_rho_3(0.9, 0.9, 0.5, 20, 12)
	rholam = my_yarn_ball_default_rho_4()

	print(rholam(0.0))
	print(rholam(1.0))

	initialize_my_yarn_ball(Nc=1, Ns=5000, nb_revs=6, nb_folds=3, rho_lambda=rholam, plot=True)

	#initialize_my_yarn_ball(Nc=1, Ns=5000, nb_revs=6, nb_folds=3, rho_lambda=my_yarn_ball_default_rho_2(0.05, 20, 0.03), plot=True)

