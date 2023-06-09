"""Naive algorithms for weighted pair counter and derivative in numba
"""
from numba import njit


@njit()
def count_weighted_pairs_3d_cpu(x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared, result):
    """Weighted pair-counter in 3 spatial dimensions

    Parameters
    ----------
    x1 : ndarray of shape (n1, )
        x-coordinate of sample 1
        n1 is typically 1e5-1e7

    y1 : ndarray of shape (n1, )
        y-coordinate of sample 1

    z1 : ndarray of shape (n1, )
        z-coordinate of sample 1

    w1 : ndarray of shape (n1, )
        weight of sample 1

    x2 : ndarray of shape (n2, )
        x-coordinate of sample 2
        n2 is typically 1e5-1e7

    y2 : ndarray of shape (n2, )
        y-coordinate of sample 2

    z2 : ndarray of shape (n2, )
        z-coordinate of sample 2

    w2 : ndarray of shape (n2, )
        weight of sample 2

    rbins_squared : ndarray of shape (nbins, )
        Squared distance of the search radii
        Algorithm assumes rbins_squared is strictly monotonically increasing
        nbins is typically 10-100

    result : ndarray of shape (nbins, )
        Empty array in which the weighted pair counts will be stored

    """
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = rbins_squared.shape[0]

    for i in range(n1):
        px = x1[i]
        py = y1[i]
        pz = z1[i]
        pw = w1[i]
        for j in range(n2):
            qx = x2[j]
            qy = y2[j]
            qz = z2[j]
            qw = w2[j]
            dx = px - qx
            dy = py - qy
            dz = pz - qz
            wprod = pw * qw
            dsq = dx * dx + dy * dy + dz * dz

            k = nbins - 1
            while dsq <= rbins_squared[k]:
                result[k - 1] += wprod
                k = k - 1
                if k <= 0:
                    break

    return result


@njit()
def count_weighted_pairs_3d_derivs_cpu(
    x1, y1, z1, w1, dw1, x2, y2, z2, w2, dw2, rbins_squared, result
):
    """Derivative of weighted pair-counter in 3 spatial dimensions

    Parameters
    ----------
    x1 : ndarray of shape (n1, )
        x-coordinate of sample 1
        n1 is typically 1e5-1e7

    y1 : ndarray of shape (n1, )
        y-coordinate of sample 1

    z1 : ndarray of shape (n1, )
        z-coordinate of sample 1

    w1 : ndarray of shape (n1, )
        weight of sample 1

    dw1 : ndarray of shape (n1, )
        Gradient of sample 1 weight w/r/t a single parameter θ

    x2 : ndarray of shape (n2, )
        x-coordinate of sample 2
        n2 is typically 1e5-1e7

    y2 : ndarray of shape (n2, )
        y-coordinate of sample 2

    z2 : ndarray of shape (n2, )
        z-coordinate of sample 2

    w2 : ndarray of shape (n2, )
        weight of sample 2

    dw2 : ndarray of shape (n2, )
        Gradient of sample 2 weight w/r/t a single parameter θ

    rbins_squared : ndarray of shape (nbins, )
        Squared distance of the search radii
        Algorithm assumes rbins_squared is strictly monotonically increasing
        nbins is typically 10-100

    result : ndarray of shape (nbins, )
        Empty array in which the weighted pair counts will be stored

    """
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    nbins = rbins_squared.shape[0]

    for i in range(n1):
        px = x1[i]
        py = y1[i]
        pz = z1[i]
        pw = w1[i]
        pdw = dw1[i]
        for j in range(n2):
            qx = x2[j]
            qy = y2[j]
            qz = z2[j]
            qw = w2[j]
            qdw = dw2[j]
            dx = px - qx
            dy = py - qy
            dz = pz - qz
            dwprod = (pw * qdw) + (qw + pdw)
            dsq = dx * dx + dy * dy + dz * dz

            k = nbins - 1
            while dsq <= rbins_squared[k]:
                result[k - 1] += dwprod
                k = k - 1
                if k <= 0:
                    break

    return result
