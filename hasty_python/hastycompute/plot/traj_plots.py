from __future__ import annotations

import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

import numpy as np

"""Display functions for trajectories."""


import itertools
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from numpy.typing import NDArray

KMAX = 0.5

DEFAULT_RESOLUTION = 6e-4  # m, i.e. 0.6 mm isotropic
DEFAULT_RASTER_TIME = 10e-3  # ms

DEFAULT_GMAX = 0.04  # T/m
DEFAULT_SMAX = 0.1  # T/m/ms

class displayConfig:
    """
    A container class used to share arguments related to display.

    The values can be updated either directy (and permanently) or temporarily by using
    a context manager.

    Examples
    --------
    >>> from mrinufft.trajectories.display import displayConfig
    >>> displayConfig.alpha
    0.2
    >>> with displayConfig(alpha=0.5):
            print(displayConfig.alpha)
    0.5
    >>> displayConfig.alpha
    0.2
    """

    alpha: float = 0.2
    """Transparency used for area plots, by default ``0.2``."""
    linewidth: float = 2
    """Width for lines or curves, by default ``2``."""
    pointsize: int = 10
    """Size for points used to show constraints, by default ``10``."""
    fontsize: int = 18
    """Font size for most labels and texts, by default ``18``."""
    small_fontsize: int = 14
    """Font size for smaller texts, by default ``14``."""
    nb_colors: int = 10
    """Number of colors to use in the color cycle, by default ``10``."""
    palette: str = "tab10"
    """Name of the color palette to use, by default ``"tab10"``.
    This can be any of the matplotlib colormaps, or a list of colors."""
    one_shot_color: str = "k"
    """Matplotlib color for the highlighted shot, by default ``"k"`` (black)."""
    one_shot_linewidth_factor: float = 2
    """Factor to multiply the linewidth of the highlighted shot, by default ``2``."""
    gradient_point_color: str = "r"
    """Matplotlib color for gradient constraint points, by default ``"r"`` (red)."""
    slewrate_point_color: str = "b"
    """Matplotlib color for slew rate constraint points, by default ``"b"`` (blue)."""

    def __init__(self, **kwargs: Any) -> None:  # noqa ANN401
        """Update the display configuration."""
        self.update(**kwargs)

    def update(self, **kwargs: Any) -> None:  # noqa ANN401
        """Update the display configuration."""
        self._old_values = {}
        for key, value in kwargs.items():
            self._old_values[key] = getattr(displayConfig, key)
            setattr(displayConfig, key, value)

    def reset(self) -> None:
        """Restore the display configuration."""
        for key, value in self._old_values.items():
            setattr(displayConfig, key, value)
        delattr(self, "_old_values")

    def __enter__(self) -> displayConfig:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: Any) -> None:  # noqa ANN401
        """Exit the context manager."""
        self.reset()

    @classmethod
    def get_colorlist(cls) -> list[str | NDArray]:
        """Extract a list of colors from a matplotlib palette.

        If the palette is continuous, the colors will be sampled from it.
        If its a categorical palette, the colors will be used in cycle.

        Parameters
        ----------
        palette : str, or list of colors, or matplotlib colormap
            Name of the palette to use, or list of colors, or matplotlib colormap.
        nb_colors : int, optional
            Number of colors to extract from the palette.
            The default is -1, and the value will be read from displayConfig.nb_colors.

        Returns
        -------
        colorlist : list of matplotlib colors.
        """
        if isinstance(cls.palette, str):
            cm = mpl.colormaps[cls.palette]
        elif isinstance(cls.palette, mpl.colors.Colormap):
            cm = cls.palette
        elif isinstance(cls.palette, list):
            cm = mpl.cm.ListedColormap(cls.palette)
        colorlist = []
        colors = getattr(cm, "colors", [])
        if 0 < len(colors) < cls.nb_colors:
            colorlist = [
                c for _, c in zip(range(cls.nb_colors), itertools.cycle(cm.colors))
            ]
        else:
            colorlist = cm(np.linspace(0, 1, cls.nb_colors))
        return colorlist

def _setup_2D_ticks(figsize: float, fig: plt.Figure | None = None, kmax: NDArray | None = None) -> plt.Axes:
    """Add ticks to 2D plot."""
    if kmax is None:
        kmax = [KMAX, KMAX]
    if fig is None:
        fig = plt.figure(figsize=(figsize, figsize))
    ax = fig if (isinstance(fig, plt.Axes)) else fig.subplots()
    ax.grid(True)
    ax.set_xticks([-kmax[0], -kmax[0] / 2, 0, kmax[0] / 2, kmax[0]])
    ax.set_yticks([-kmax[1], -kmax[1] / 2, 0, kmax[1] / 2, kmax[1]])
    ax.set_xlim((-kmax[0], kmax[0]))
    ax.set_ylim((-kmax[1], kmax[1]))
    ax.set_xlabel("kx", fontsize=displayConfig.fontsize)
    ax.set_ylabel("ky", fontsize=displayConfig.fontsize)
    return ax


def _setup_3D_ticks(figsize: float, fig: plt.Figure | None = None, kmax: NDArray | None = None) -> plt.Axes:
    """Add ticks to 3D plot."""
    if kmax is None:
        kmax = [KMAX, KMAX, KMAX]
    if fig is None:
        fig = plt.figure(figsize=(figsize, figsize))
    ax = fig if (isinstance(fig, plt.Axes)) else fig.add_subplot(projection="3d")
    ax.set_xticks([-kmax[0], -kmax[0] / 2, 0, kmax[0] / 2, kmax[0]])
    ax.set_yticks([-kmax[1], -kmax[1] / 2, 0, kmax[1] / 2, kmax[1]])
    ax.set_zticks([-kmax[2], -kmax[2] / 2, 0, kmax[2] / 2, kmax[2]])
    ax.axes.set_xlim3d(left=-kmax[0], right=kmax[0])
    ax.axes.set_ylim3d(bottom=-kmax[1], top=kmax[1])
    ax.axes.set_zlim3d(bottom=-kmax[2], top=kmax[2])
    ax.set_box_aspect((2 * kmax[0], 2 * kmax[1], 2 * kmax[2]))
    ax.set_xlabel("kx", fontsize=displayConfig.fontsize)
    ax.set_ylabel("ky", fontsize=displayConfig.fontsize)
    ax.set_zlabel("kz", fontsize=displayConfig.fontsize)
    return ax

def display_2D_trajectory(
    trajectory: NDArray,
    figsize: float = 5,
    one_shot: bool | int = False,
    subfigure: plt.Figure | plt.Axes | None = None,
    kmax: NDArray | None = None
) -> plt.Axes:
    """Display 2D trajectories.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory to display.
    figsize : float, optional
        Size of the figure.
    one_shot : bool or int, optional
        State if a specific shot should be highlighted in bold black.
        If `True`, highlight the middle shot.
        If `int`, highlight the shot at that index.
        The default is `False`.
    subfigure: plt.Figure, plt.SubFigure or plt.Axes, optional
        The figure where the trajectory should be displayed.
        The default is `None`.

    Returns
    -------
    ax : plt.Axes
        Axes of the figure.
    """
    # Setup figure and ticks
    Nc, Ns = trajectory.shape[:2]
    ax = _setup_2D_ticks(figsize, subfigure, kmax)
    colors = displayConfig.get_colorlist()
    # Display every shot
    for i in range(Nc):
        ax.plot(
            trajectory[i, :, 0],
            trajectory[i, :, 1],
            color=colors[i % displayConfig.nb_colors],
            linewidth=displayConfig.linewidth,
        )

    # Display one shot in particular if requested
    if one_shot is not False:  # If True or int
        # Select shot
        shot_id = Nc // 2
        if one_shot is not True:  # If int
            shot_id = one_shot

        # Highlight the shot in black
        ax.plot(
            trajectory[shot_id, :, 0],
            trajectory[shot_id, :, 1],
            color=displayConfig.one_shot_color,
            linewidth=displayConfig.one_shot_linewidth_factor * displayConfig.linewidth,
        )

    return ax

def display_3D_trajectory(
    trajectory: NDArray,
    nb_repetitions: int | None = None,
    figsize: float = 5,
    per_plane: bool = True,
    one_shot: bool | int = False,
    subfigure: plt.Figure | plt.Axes | None = None,
    kmax: NDArray | None = None
) -> plt.Axes:
    """Display 3D trajectories.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory to display.
    nb_repetitions : int
        Number of repetitions (planes, cones, shells, etc).
        The default is `None`.
    figsize : float, optional
        Size of the figure.
    per_plane : bool, optional
        If True, display the trajectory with a different color
        for each plane.
    one_shot : bool or int, optional
        State if a specific shot should be highlighted in bold black.
        If `True`, highlight the middle shot.
        If `int`, highlight the shot at that index.
        The default is `False`.
    subfigure: plt.Figure, plt.SubFigure or plt.Axes, optional
        The figure where the trajectory should be displayed.
        The default is `None`.
    Returns
    -------
    ax : plt.Axes
        Axes of the figure.
    """
    # Setup figure and ticks, and handle 2D trajectories
    ax = _setup_3D_ticks(figsize, subfigure, kmax)
    if nb_repetitions is None:
        nb_repetitions = trajectory.shape[0]
    if trajectory.shape[-1] == 2:
        trajectory = np.concatenate(
            [trajectory, np.zeros((*(trajectory.shape[:2]), 1))], axis=-1
        )
    trajectory = trajectory.reshape((nb_repetitions, -1, trajectory.shape[-2], 3))
    Nc, Ns = trajectory.shape[1:3]

    colors = displayConfig.get_colorlist()
    # Display every shot
    for i in range(nb_repetitions):
        for j in range(Nc):
            ax.plot(
                trajectory[i, j, :, 0],
                trajectory[i, j, :, 1],
                trajectory[i, j, :, 2],
                color=colors[(i + j * (not per_plane)) % displayConfig.nb_colors],
                linewidth=displayConfig.linewidth,
            )

    # Display one shot in particular if requested
    if one_shot is not False:  # If True or int
        trajectory = trajectory.reshape((-1, Ns, 3))

        # Select shot
        shot_id = Nc // 2
        if one_shot is not True:  # If int
            shot_id = one_shot

        # Highlight the shot in black
        ax.plot(
            trajectory[shot_id, :, 0],
            trajectory[shot_id, :, 1],
            trajectory[shot_id, :, 2],
            color=displayConfig.one_shot_color,
            linewidth=displayConfig.one_shot_linewidth_factor * displayConfig.linewidth,
        )
        trajectory = trajectory.reshape((-1, Nc, Ns, 3))

    return ax


def show_trajectory(trajectory, one_shot, figure_size, kmax=None):
    if trajectory.shape[-1] == 2:
        ax = display_2D_trajectory(
            trajectory, 
            figsize=figure_size, 
            one_shot=one_shot % trajectory.shape[0],
            kmax=kmax
        )
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()
    else:
        ax = display_3D_trajectory(
            trajectory,
            figsize=figure_size,
            one_shot=one_shot % trajectory.shape[0],
            per_plane=False,
            kmax=kmax,
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.show()


def show_trajectories(
    function, arguments, one_shot, subfig_size, dim="3D", axes=(0, 1)
):
    # Initialize trajectories with varying option
    trajectories = [function(arg) for arg in arguments]

    # Plot the trajectories side by side
    fig = plt.figure(
        figsize=(len(trajectories) * subfig_size, subfig_size),
        constrained_layout=True,
    )
    subfigs = fig.subfigures(1, len(trajectories), wspace=0)
    for subfig, arg, traj in zip(subfigs, arguments, trajectories):
        if dim == "3D" and traj.shape[-1] == 3:
            ax = display_3D_trajectory(
                traj,
                size=subfig_size,
                one_shot=one_shot % traj.shape[0],
                subfigure=subfig,
                per_plane=False,
            )
        else:
            ax = display_2D_trajectory(
                traj[..., axes],
                size=subfig_size,
                one_shot=one_shot % traj.shape[0],
                subfigure=subfig,
            )
        labels = ["kx", "ky", "kz"]
        ax.set_xlabel(labels[axes[0]], fontsize=displayConfig.fontsize)
        ax.set_ylabel(labels[axes[1]], fontsize=displayConfig.fontsize)
        ax.set_aspect("equal")
        ax.set_title(str(arg), fontsize=displayConfig.fontsize)
    plt.show()


def show_density(density, figure_size, *, log_scale=False):
    density = density.T[::-1]

    plt.figure(figsize=(figure_size, figure_size))
    if log_scale:
        plt.imshow(density, cmap="jet", norm=colors.LogNorm())
    else:
        plt.imshow(density, cmap="jet")

    ax = plt.gca()
    ax.set_xlabel("kx", fontsize=displayConfig.fontsize)
    ax.set_ylabel("ky", fontsize=displayConfig.fontsize)
    ax.set_aspect("equal")

    plt.axis(False)
    plt.colorbar()
    plt.show()


def show_densities(function, arguments, subfig_size, *, log_scale=False):
    # Initialize k-space densities with varying option
    densities = [function(arg).T[::-1] for arg in arguments]

    # Plot the trajectories side by side
    fig, axes = plt.subplots(
        1,
        len(densities),
        figsize=(len(densities) * subfig_size, subfig_size),
        constrained_layout=True,
    )

    for ax, arg, density in zip(axes, arguments, densities):
        ax.set_title(str(arg), fontsize=displayConfig.fontsize)
        ax.set_xlabel("kx", fontsize=displayConfig.fontsize)
        ax.set_ylabel("ky", fontsize=displayConfig.fontsize)
        ax.set_aspect("equal")
        if log_scale:
            ax.imshow(density, cmap="jet", norm=colors.LogNorm())
        else:
            ax.imshow(density, cmap="jet")
        ax.axis(False)
    plt.show()


def show_locations(function, arguments, subfig_size, *, log_scale=False):
    # Initialize k-space locations with varying option
    locations = [function(arg) for arg in arguments]

    # Plot the trajectories side by side
    fig, axes = plt.subplots(
        1,
        len(locations),
        figsize=(len(locations) * subfig_size, subfig_size),
        constrained_layout=True,
    )

    for ax, arg, location in zip(axes, arguments, locations):
        ax.set_title(str(arg), fontsize=displayConfig.fontsize)
        ax.set_xlim(-1.05 * KMAX, 1.05 * KMAX)
        ax.set_ylim(-1.05 * KMAX, 1.05 * KMAX)
        ax.set_xlabel("kx", fontsize=displayConfig.fontsize)
        ax.set_ylabel("ky", fontsize=displayConfig.fontsize)
        ax.set_aspect("equal")
        ax.scatter(location[..., 0], location[..., 1], s=3)
    plt.show()

