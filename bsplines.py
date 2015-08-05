from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import splev


def universal_basis(boundaries, degree, dimension):
    """Create a universal B-spline basis.

    A universal B-spline basis has uniformly spaced interior
    knots. This function determines the number of interior knots to
    include using the `dimension` argument.

    # Arguments

    boundaries : A 2-tuple containing floats (low, high).
    degree : The degree of the B-spline's polynomial pieces.
    dimension : The number of columns in the feature matrix.

    # Returns

    A `BSplineBasis` object.

    """
    num_chunks = int(dimension) - int(degree)
    lo, hi = boundaries
    inner_knots = np.linspace(lo, hi, num_chunks + 1)
    return BSplineBasis(boundaries, inner_knots, degree)


class BSplineBasis:
    def __init__(self, boundaries, inner_knots, degree):
        self.boundaries = boundaries
        self.inner_knots = inner_knots
        self.degree = degree
        self.tck = basis_tck(boundaries, inner_knots, degree)

    @property
    def dimension(self):
        """The number of columns in the feature matrix."""
        knots, _, degree = self.tck
        return num_basis_funcs(knots, degree)

    def eval(self, x):
        """Evaluate the B-spline bases to construct a feature matrix.

        # Arguments

        x : The points at which the bases are evaluated.

        # Returns

        A feature matrix containing the basis function values in each
        column.

        """
        f = splev(x, self.tck)
        B = np.asarray(f).T
        return B

    def plot(self, ndx=100, *args, **kwargs):
        """Plot the B-spline basis functions."""
        lo, hi = self.boundaries
        x = np.linspace(lo, hi, ndx)
        B = self.eval(x)

        fig, ax = plt.subplots(1, 1, *args, **kwargs)
        for y in B.T:
            ax.plot(x, y)

        return fig


def basis_tck(boundaries, inner_knots, degree):
    """Compute the tck B-spline basis representation.

    The tck representation is 3-tuple containing the complete knots
    (t), the coefficients parameterizing the basis (c), and the degree
    of the polynomial pieces (k).

    To parameterize a single B-spline, we would use a vector for the
    coefficients. In this case, however, we want to represent the
    whole basis and so we use an identity matrix with the same number
    of rows and columns as the number of basis functions. This
    effectively represents a collection of B-splines that are exactly
    the bases (each row contains a single 1-valued positive
    coefficient).

    # Arguments

    boundaries  : A tuple containing two floats (low, high).
    inner_knots : A sequence of interior B-spline knots.
    degree      : The degree of the B-spline pieces.

    # Returns

    A 3-tuple containing the complete knots (t), the basis
    coefficients (c), and the degree of the polynomial pieces (k).

    """
    t = complete_knots(boundaries, inner_knots, degree)
    c = np.eye(num_basis_funcs(t, degree))
    k = int(degree)
    return t, c, k


def complete_knots(boundaries, inner_knots, degree):
    """Construct the full B-spline knot sequence.

    # Arguments

    boundaries  : A tuple containing two floats (low, high).
    inner_knots : A sequence of interior B-spline knots.
    degree      : The degree of the B-spline pieces.

    # Returns

    A numpy array containing the full B-spline knot sequence.

    """
    lower, upper = boundaries_to_knots(boundaries, degree)
    knots = np.r_[lower, inner_knots, upper]
    return knots


def boundaries_to_knots(boundaries, degree):
    """Construct the knot sequences used at the boundaries of a B-spline.

    # Arguments

    boundaries : A tuple containing two floats (low, high).
    degree     : The degree of the B-spline pieces.

    # Returns

    A 2-tuple containing the lower and upper knot sequences as lists.

    """
    d = int(degree)
    lo, hi = boundaries
    return d * [lo], d * [hi]


def num_basis_funcs(knots, degree):
    """Compute the number of bases determined by the knots and degree.

    # Arguments

    knots : The complete B-spline knot sequence.
    degree : The degree of the B-spline's polynomial pieces.

    # Returns

    The number of basis functions used to represent the B-spline. That
    is, the dimension of the basis expansion at a given input point.

    """
    return len(knots) - degree_to_order(degree)



def degree_to_order(degree):
    """Compute the order given the degree of a B-spline."""
    return int(degree) + 1


def order_to_degree(order):
    """Compute the degree given the order of a B-spline."""
    return int(order) - 1
