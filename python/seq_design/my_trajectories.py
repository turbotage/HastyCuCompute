"""Functions to initialize 3D trajectories."""

from functools import partial
from typing import Literal

import numpy as np
import numpy.linalg as nl
from numpy.typing import NDArray
from scipy.special import ellipj, ellipk

from mrinufft.trajectories.maths import (
    CIRCLE_PACKING_DENSITY,
    EIGENVECTOR_2D_FIBONACCI,
    R2D,
    Ra,
    Ry,
    Rz,
    generate_fibonacci_circle
)

from mrinufft.trajectories.tools import conify, duplicate_along_axes, epify, precess, stack
from mrinufft.trajectories.trajectory2D import initialize_2D_spiral, initialize_2D_radial




def initialize_sloped_3D_cones(
    Nc: int,
    Ns: int,
    tilt: str | float = "golden",
    in_out: bool = False,
    nb_zigzags: float = 5,
    spiral: str | float = "archimedes",
    width: float = 1,
    ) -> NDArray:

    single_spiral = initialize_2D_spiral(
        Nc=1, 
        Ns=Ns, 
        spiral=spiral, 
        nb_revolutions=nb_zigzags
    )

    # Estimate best cone angle based on the ratio between
    # sphere volume divided by Nc and spherical sector packing optimaly a sphere
    max_angle = np.pi / 2 - width * np.arccos(
        1 - CIRCLE_PACKING_DENSITY * 2 / Nc / (1 + in_out)
    )


