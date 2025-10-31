import torch
import torch.nn.functional as F

def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor, dim: int=-1, extrapolate: str='constant') -> torch.Tensor:
	"""One-dimensional linear interpolation between monotonically increasing sample
	points, with extrapolation beyond sample points.

	Returns the one-dimensional piecewise linear interpolant to a function with
	given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

	Args:
		x: The :math:`x`-coordinates at which to evaluate the interpolated
			values.
		xp: The :math:`x`-coordinates of the data points, must be increasing.
		fp: The :math:`y`-coordinates of the data points, same shape as `xp`.
		dim: Dimension across which to interpolate.
		extrapolate: How to handle values outside the range of `xp`. Options are:
			- 'linear': Extrapolate linearly beyond range of xp values.
			- 'constant': Use the boundary value of `fp` for `x` values outside `xp`.

	Returns:
		The interpolated values, same size as `x`.
	"""
	# Move the interpolation dimension to the last axis
	x = x.movedim(dim, -1)
	xp = xp.movedim(dim, -1)
	fp = fp.movedim(dim, -1)
	
	m = torch.diff(fp) / torch.diff(xp) # slope
	b = fp[..., :-1] - m * xp[..., :-1] # offset
	indices = torch.searchsorted(xp, x, right=False)
	
	if extrapolate == 'constant':
		# Pad m and b to get constant values outside of xp range
		m = torch.cat([torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1)
		b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
	else: # extrapolate == 'linear'
		indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

	values = m.gather(-1, indices) * x + b.gather(-1, indices)
	
	return values.movedim(-1, dim)

def catmull_rom_interp(x, num_steps):
    """
    Catmull-Rom cubic interpolation along last axis.
    Args:
        x: (N, C, L) tensor of control points
        num_steps: int, number of output samples along the time axis
    Returns:
        (N, C, num_steps) tensor (same dtype & device as x)
    """
    N, C, L = x.shape
    device = x.device
    dtype = x.dtype

    # sample positions in control-index space [0, L-1]
    t = torch.linspace(0.0, float(L - 1), num_steps, device=device, dtype=dtype)  # (M,)
    t0 = torch.floor(t).long()                                                    # (M,)
    frac = (t - t0.to(dtype))                                                    # (M,)

    # indices of neighbors: i-1, i, i+1, i+2 (clamped to valid range)
    i_m1 = (t0 - 1).clamp(min=0, max=L-1)   # (M,)
    i_0  = t0.clamp(min=0, max=L-1)
    i_p1 = (t0 + 1).clamp(min=0, max=L-1)
    i_p2 = (t0 + 2).clamp(min=0, max=L-1)

    # expand indices to (N, C, M) for gather
    # first make (1,1,M) then expand
    idx_shape = (1, 1, num_steps)
    im1 = i_m1.view(*idx_shape).expand(N, C, -1)
    i0b = i_0.view(*idx_shape).expand(N, C, -1)
    ip1 = i_p1.view(*idx_shape).expand(N, C, -1)
    ip2 = i_p2.view(*idx_shape).expand(N, C, -1)

    # gather p0..p3
    # torch.gather requires indices of type Long and the same dims as input
    p0 = torch.gather(x, dim=2, index=im1)   # (N, C, M)
    p1 = torch.gather(x, dim=2, index=i0b)
    p2 = torch.gather(x, dim=2, index=ip1)
    p3 = torch.gather(x, dim=2, index=ip2)

    # reshape frac to (1,1,M) for broadcasting
    t_ = frac.view(1, 1, -1).to(dtype)

    # Catmull-Rom basis (standard formulation)
    a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
    a1 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3
    a2 = -0.5 * p0 + 0.5 * p2
    a3 = p1

    # Evaluate cubic polynomial per sample
    result = ((a0 * t_ + a1) * t_ + a2) * t_ + a3   # (N, C, M)

    return result


def lagged_convolve(input, kernel, left_pad_value=0.0, right_pad_value=0.0):
	"""Convolve input with kernel using lagged method to avoid boundary effects.

	Args:
		input: Input tensor of shape (..., N).
		kernel: Convolution kernel of shape (K,).

	Returns:
		Convolved tensor of shape (..., N + K - 1).
	"""
	K = kernel.shape[0]
	pad = K - 1
	# Pad input at the beginning with zeros
	padded_input = torch.nn.functional.pad(input, (pad, 0), value=left_pad_value)
	padded_input = torch.nn.functional.pad(padded_input, (0, pad), value=right_pad_value)
	# Perform convolution
	convolved = torch.nn.functional.conv1d(
		padded_input.unsqueeze(0) if padded_input.dim() == 1 else padded_input, 
		kernel.view(1, 1, -1), 
		groups=1
	).squeeze(0)
	return convolved

def convolve(input, kernel, left_pad_value=0.0, right_pad_value=0.0):
	"""Convolve input with kernel using lagged method to avoid boundary effects.

	Args:
		input: Input tensor of shape (..., N).
		kernel: Convolution kernel of shape (K,).

	Returns:
		Convolved tensor of shape (..., N + K - 1).
	"""
	K = kernel.shape[0]
	pad = K - 1
	# Pad input at the beginning with zeros
	padded_input = torch.nn.functional.pad(input, (pad, 0), value=left_pad_value)
	#padded_input = torch.nn.functional.pad(padded_input, (0, pad), value=right_pad_value)
	# Perform convolution
	convolved = torch.nn.functional.conv1d(
		padded_input.unsqueeze(0) if padded_input.dim() == 1 else padded_input, 
		kernel.view(1, 1, -1), 
		groups=1
	).squeeze(0)
	return convolved

def formated_list_print(lst, fmt="{:.4e}"):
	"""Print a list of numbers in formatted way."""
	formatted = [fmt.format(x) for x in lst]
	return "[ " + ", ".join(formatted) + " ]"

if __name__ == "__main__":
	# Example usage

	if False:
		x = torch.tensor([0.5, 1.5, 2.5])
		xp = torch.tensor([0.0, 1.0, 2.0, 3.0])
		fp = torch.tensor([0.0, 1.0, 0.0, 1.0])
		interpolated_values = interp(x, xp, fp)
		print(interpolated_values)  # Should print interpolated values at x

		input_signal = torch.tensor([1.0, 2.0, 3.0])
		kernel = torch.tensor([0.2, 0.5, 0.3])
		convolved_signal = lagged_convolve(input_signal, kernel)
		print(convolved_signal)  # Should print the convolved signal

	if True:
		t = torch.linspace(0,1,10).detach()
		x = torch.sin(4*torch.pi*t).expand(5,1,-1).detach()

		tup = torch.linspace(0,1,100).detach()
		xup = catmull_rom_interp(x, 100).detach().squeeze()

		import matplotlib.pyplot as plt
		plt.figure()
		plt.plot(t.cpu().numpy(), x[0,0,...].cpu().numpy(), 'o-', label='Original')
		plt.plot(tup.cpu().numpy(), xup[0,...].cpu().numpy(), '-', label='Cubic Interpolated')
		plt.legend()
		plt.show()

