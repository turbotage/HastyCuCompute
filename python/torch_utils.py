import torch

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
    x = torch.tensor([0.5, 1.5, 2.5])
    xp = torch.tensor([0.0, 1.0, 2.0, 3.0])
    fp = torch.tensor([0.0, 1.0, 0.0, 1.0])
    interpolated_values = interp(x, xp, fp)
    print(interpolated_values)  # Should print interpolated values at x

    input_signal = torch.tensor([1.0, 2.0, 3.0])
    kernel = torch.tensor([0.2, 0.5, 0.3])
    convolved_signal = lagged_convolve(input_signal, kernel)
    print(convolved_signal)  # Should print the convolved signal