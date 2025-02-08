import os
import sys
import mrinufft as mn
import numpy as np
import matplotlib.pyplot as plt
import math

import pypulseq as pp

import scipy as sp
from scipy.interpolate import CubicSpline
import scipy.special as sps


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traj_utils as tu

def plot31(gcurve, title=''):
	plt.figure(1)
	plt.plot(gcurve[0, :], 'r-*')
	plt.plot(gcurve[1, :], 'g-*')
	plt.plot(gcurve[2, :], 'b-*')
	plt.title(title)
	plt.show()




#shape_3d = (256, 256, 256)
shape_3d = (64,64,32)

# Display parameters
figure_size = 5.5  # Figure size for trajectory plots
subfigure_size = 3  # Figure size for subplots
one_shot = 0  # Highlight one shot in particular


# arguments = [0, 0.1, 0.2, 0.3]
# function = lambda x: mn.create_cutoff_decay_density(
#     shape=shape_3d,
#     cutoff=x,
#     decay=2,
# )[shape_3d[0] // 2, :, :]
# tu.show_densities(
#     function,
#     arguments,
#     subfig_size=subfigure_size,
# )

density = mn.create_cutoff_decay_density(shape=shape_3d, cutoff=0.2, decay=2)


# trajectory = mn.oversample(
#     mn.initialize_3D_travelling_salesman(Nc, Ns, density=density, method="lloyd", verbose=True), 
#     4*Ns
# )

samp_method = "se-seiffert"
#samp_method = "se-floret"
#samp_method = "se-random"
#samp_method = "tsp"

if samp_method == "tsp":
	clustering=("phi", "theta", "r")

	# Trajectory parameters
	Nc = 50  # Number of shots
	Ns = 10  # Number of samples per shot

	undersamp_traj = np.zeros((Nc, Ns+2, 3))
	undersamp_traj[:,2:,:] = mn.initialize_3D_travelling_salesman(Nc, Ns, density=density, 
		first_cluster_by=clustering[0],
		second_cluster_by=clustering[1],
		sort_by=clustering[2],                           
		verbose=True
	)
elif samp_method == "se-seiffert":
	# Trajectory parameters
	Nc = 50  # Number of shots
	Ns = 50  # Number of samples per shot

	crv_idx = 0.9999

	undersamp_traj = np.zeros((Nc, Ns+2, 3))
	undersamp_traj[:,2:,:] = mn.initialize_3D_seiffert_spiral(Nc, Ns, curve_index=crv_idx, in_out=False)
elif samp_method == "se-floret":
	# Trajectory parameters
	Nc = 50  # Number of shots
	Ns = 50  # Number of samples per shot

	nbrev = 5

	undersamp_traj = np.zeros((Nc, Ns+2, 3))
	undersamp_traj[:,2:,:] = mn.initialize_3D_floret(Nc, Ns, nb_revolutions=nbrev, 
						spiral="fermat", cone_tilt="golden", in_out=False)
elif samp_method == "se-random":
	# Trajectory parameters
	Nc = 50  # Number of shots
	Ns = 20

	undersamp_traj = np.zeros((Nc, Ns+2, 3))
	undersamp_traj[:,2:,:] = mn.initialize_3D_random_walk(Nc, Ns, density)



system = pp.Opts(max_grad=80, grad_unit='mT/m', max_slew=200, slew_unit='T/m/s', B0=3.0, grad_raster_time=10e-6)
seq = pp.Sequence(system=system)



fov = 220e-3
resolution = 384
delta_k = 1.0 / fov

g = []
s = []

tmax = 1e-3

plot_slew_grad_v = False
safe_gradients = False
for ti in range(100):

	slew_failure = False
	grad_failure = False

	nsamp = int(tmax / system.grad_raster_time)
	trajectory = mn.oversample(
		undersamp_traj, 
		nsamp #nsamp
	) * (resolution * delta_k)

	max_grad = 0
	max_slew = 0
	g.clear()
	s.clear()
	for i in range(Nc):
		gi, si = pp.traj_to_grad(trajectory[i,...].transpose(1,0), raster_time=system.grad_raster_time)

		max_grad = np.abs(gi).max()
		max_slew = np.abs(si).max()
		if max_grad > system.max_grad:
			print('Gradient failure: ', max_grad)
			grad_failure = True
			break
		elif max_slew > system.max_slew:
			print('Slew failure: ', max_grad)
			slew_failure = True
			break

		if i == 0 and plot_slew_grad_v:
			plot31(si, 'Slew Rate')
			plot31(gi, 'Gradient')

			v = np.sqrt(np.sum(np.square(gi), axis=0))

			plt.figure()
			plt.plot(v)
			plt.title('K-space speed')
			plt.show()

		g.append(gi)
		s.append(si)

	if grad_failure:
		tmax *= (1.05 * max_grad / system.max_grad)
	elif slew_failure:
		tmax *= (1.05 * max_slew / system.max_slew)
	else:
		safe_gradients = True
		break

if not safe_gradients:
	raise RuntimeError("Gradient or slew rate failure")

print('Aquisition Duration: ', tmax * 1000.0, ' ms')


for i in range(0, Nc, 50):
    tu.show_trajectory(0.5*trajectory / np.abs(trajectory).max(), figure_size=figure_size, one_shot=i)



g = np.stack(g, axis=0)
s = np.stack(s, axis=0)

print('Current Slew Rate: ', np.max(np.abs(s)), 'System Max: ', system.max_slew)

gx = []
gy = []
gz = []

for i in range(Nc):
	gx.append(pp.make_arbitrary_grad(channel='x', waveform=g[i, 0, :], system=system))
	gy.append(pp.make_arbitrary_grad(channel='y', waveform=g[i, 1, :], system=system))
	gz.append(pp.make_arbitrary_grad(channel='z', waveform=g[i, 2, :], system=system))
      
# Velocity encodings

test_trap1 = pp.make_trapezoid(channel='z', system=system, area=0.5, duration=1e-3)
grad_height = test_trap1.flat_area / test_trap1.flat_time
test_trap1 = pp.points_to_waveform(
	np.array([0.0, grad_height, grad_height, 0.0]), 
	system.grad_raster_time, 
	np.array([0.0, test_trap1.rise_time, test_trap1.flat_time, test_trap1.fall_time])
)
test_trap2 = pp.make_trapezoid(channel='z', system=system, area=-0.5, duration=1e-3)
grad_height = test_trap2.flat_area / test_trap2.flat_time
test_trap2 = pp.points_to_waveform(
	np.array([0.0, grad_height, grad_height, 0.0]), 
	system.grad_raster_time, 
	np.array([0.0, test_trap2.rise_time, test_trap2.flat_time, test_trap2.fall_time])
)

grad = np.concatenate([test_trap1, test_trap2])

      