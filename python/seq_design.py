import numpy as np
import matplotlib.pyplot as plt
import math

import pypulseq as pp

def scatter_3d(coord, marker='.', markersize=1 ,title='', axis_labels=['X Label', 'Y Label', 'Z Label']):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	
	ax.scatter(coord[0,:], coord[1,:], coord[2,:], marker=marker, s=markersize)
	ax.set_xlabel(axis_labels[0])
	ax.set_ylabel(axis_labels[1])
	ax.set_zlabel(axis_labels[2])

	plt.title(title)

	plt.show()

def get_base_spoke(wanted_readout_time, coeffs, type, system, plot=False):

    raster_points = math.ceil(wanted_readout_time / system.grad_raster_time)
    max_readout_time = raster_points * system.grad_raster_time

    def do_one_spoke(tr):

        kp = np.zeros((3, raster_points))
        
        readout_times = np.linspace(0, max_readout_time, raster_points)

        #if type == 'cone_1':
        #    w_0 = tr * np.pi / np.sqrt(max_readout_time) 
        #    kp[0, :] = coeffs[1] * np.sqrt(readout_times) * np.cos(w_0 * np.sqrt(readout_times))
        #    kp[1, :] = coeffs[1] * np.sqrt(readout_times) * np.sin(w_0 * np.sqrt(readout_times))
        #    kp[2, :] = coeffs[0] * np.sqrt(readout_times)
        if type == 'cone_1':
            #w_0 = tr * np.pi / np.log(1 + max_readout_time)
            #kp[0, :] = coeffs[1] * (readout_times) * np.cos(w_0 * np.log(1 + readout_times))
            #kp[1, :] = coeffs[1] * (readout_times) * np.sin(w_0 * np.log(1 + readout_times))
            #kp[2, :] = coeffs[0] * (readout_times)

            w_0 = tr * np.pi / max_readout_time
            kp[0, :] = coeffs[1] * readout_times * np.cos(w_0 * readout_times)
            kp[1, :] = coeffs[1] * readout_times * np.sin(w_0 * readout_times)
            kp[2, :] = coeffs[0] * (readout_times) * np.exp(-(0.1 / max_readout_time) * readout_times)

        grad, slew = pp.traj_to_grad(kp, system.grad_raster_time)

        normalize_vel = True
        if normalize_vel:
            vel = np.sqrt(np.sum(grad*grad, axis=0))
            vel /= vel[0]
            vel_last = vel[-1] + (vel[-1] - vel[-2])
            vel = np.concatenate([vel, vel_last[None]])

            plt.figure()
            plt.plot(readout_times, vel, 'r-*')
            plt.title('Velocity')
            plt.show()

            kp /= np.sqrt(vel[None,:])

        grad, slew = pp.traj_to_grad(kp, system.grad_raster_time)
        vel = np.sqrt(np.sum(grad*grad, axis=0))
        vel /= vel[0]
        vel_last = vel[-1] + (vel[-1] - vel[-2])
        vel = np.concatenate([vel, vel_last[None]])

        if plot:
            scatter_skip = max(1,raster_points // 1e4)
            scatter_3d(kp[:, 0::scatter_skip], title='kp')

            plt.figure()
            plt.plot(readout_times, vel, 'r-*')
            plt.title('Velocity new')
            plt.show()

        return kp,grad,slew
    
    best_turn_rate = 0.0
    turn_rate = 10.0
    iter = 0
    while iter < 25:
        iter += 1
        if turn_rate < 0.5:
            raise RuntimeError("Turn rate became less that half a turn per readout time, for this width and max_slew_rate")
        
        kpoints, grad, slew = do_one_spoke(turn_rate)

        max_slew = math.ceil(np.max(np.abs(slew)).item())
            
        print('Max Slew Rate: ', max_slew, 'System Max: ', system.max_slew)

        if np.sqrt(3)*np.max(slew) >= 0.9*system.max_slew:
            if math.ceil(np.max(np.abs(slew[2,:])).item()) == max_slew:
                raise RuntimeError('Slew Rate in Z is limiting, turn rate wont help')
            turn_rate *= 0.85
        else:
            if turn_rate > best_turn_rate:
                best_turn_rate = turn_rate
            turn_rate *= 1.15

        # If we take less than 20 samples per turn, there is no need to increase the turn rate,
        # what we should do is increase the sample rate
        # This is an average, of course 20 samples per turn is alot right at the start of the cone
        if raster_points / turn_rate < 20:
            print('Less then 20 samples per turn, breaking')
            break

    kpoints, grad, slew = do_one_spoke(best_turn_rate)
    
    max_slew = math.ceil(np.max(np.abs(slew)).item())
    print('Final Max Slew Rate: ', max_slew, 'System Max: ', system.max_slew)

    return kpoints, grad, slew
        

    


system = pp.Opts(max_grad=80, grad_unit='mT/m', max_slew=200, slew_unit='T/m/s')
seq = pp.Sequence(system=system)

fov = 220e-3
delta_k = 1 / fov
resolution = 320
kmax = delta_k * resolution / 2.0
max_tread = 5e-3

print('kmax: ', kmax, ' [1/m]')

coeffs = [
    kmax/np.log(1+max_tread),
    0.25 * kmax / np.sqrt(max_tread)
]

spoke_type = 'cone_1'

base_cone_kp, base_cone_grad, base_cone_slew = get_base_spoke(wanted_readout_time=max_tread, coeffs=coeffs, 
                                                    type=spoke_type, system=system, plot=True)

gx = pp.make_arbitrary_grad(channel='x', waveform=base_cone_grad[0,:], system=system)
gy = pp.make_arbitrary_grad(channel='y', waveform=base_cone_grad[1,:], system=system)
gz = pp.make_arbitrary_grad(channel='z', waveform=base_cone_grad[2,:], system=system)

print('Hello')

#def calculate_traj(kdir, nsamp, base_cone=None):
