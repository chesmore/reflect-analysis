import numpy as np
from numpy import random as rand
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class field:
    N = 9  # Multiplication factor of the harmonic
    # mixers used in data acquisition.
    # In the W(D)-band (75-125GHz), N=9(12).

    ang_inc_deg = 10  # angle of incidence of the source and receiver [deg]
    pol = "TM"  # polarization of the source signal
    c = 300.0  # speed of light [mm GHz]


def get_transfer_matrix_ang_inc(N_ar, D_ar, freq):

    ang_new = field.ang_inc_deg * np.pi / 180.0  # Angle of incidence [rad]

    if field.pol != "TM" and field.pol != "TE":
        print("Invalid Polarization.  Please use TE or TM.")

    layers = len(D_ar)  # Number of layers in the structure

    # Get angles of incidence for each layer (Snell's law)
    ang_inc_ar = []
    ang_inc_ar.append(ang_new)
    for i in range(1, layers):
        ang_new = np.arcsin(N_ar[i - 1].real / N_ar[i].real * np.sin(ang_new))
        ang_inc_ar.append(ang_new)
    ang_inc_ar = np.array(ang_inc_ar)

    # Effective thickness of the layers
    D_eff = D_ar / np.cos(ang_inc_ar)

    # Initialize the transfermatrix as the identity (complex)
    TM = np.identity(2, dtype=complex)

    # Create the transfer matrix
    for i in range(layers - 1, 0, -1):

        tm = np.ones((2, 2), dtype=complex)

        if field.pol == "TM":
            tm[0, 0] = N_ar[i] / N_ar[i - 1] + np.cos(ang_inc_ar[i]) / np.cos(
                ang_inc_ar[i - 1]
            )
            tm[0, 1] = N_ar[i] / N_ar[i - 1] - np.cos(ang_inc_ar[i]) / np.cos(
                ang_inc_ar[i - 1]
            )
            tm[1, 0] = N_ar[i] / N_ar[i - 1] - np.cos(ang_inc_ar[i]) / np.cos(
                ang_inc_ar[i - 1]
            )
            tm[1, 1] = N_ar[i] / N_ar[i - 1] + np.cos(ang_inc_ar[i]) / np.cos(
                ang_inc_ar[i - 1]
            )

        if field.pol == "TE":
            tm[0, 0] = 1 + N_ar[i] / N_ar[i - 1] * np.cos(ang_inc_ar[i]) / np.cos(
                ang_inc_ar[i - 1]
            )
            tm[0, 1] = 1 - N_ar[i] / N_ar[i - 1] * np.cos(ang_inc_ar[i]) / np.cos(
                ang_inc_ar[i - 1]
            )
            tm[1, 0] = 1 - N_ar[i] / N_ar[i - 1] * np.cos(ang_inc_ar[i]) / np.cos(
                ang_inc_ar[i - 1]
            )
            tm[1, 1] = 1 + N_ar[i] / N_ar[i - 1] * np.cos(ang_inc_ar[i]) / np.cos(
                ang_inc_ar[i - 1]
            )

        tm = 0.5 * tm

        # Dotted layer into the total transfer matrix.
        TM = np.dot(tm, TM)

        # Add propogation matrix.
        # This adds the phase delay for the layer.

        prop = np.identity(2, dtype=complex)

        ptx = 2.0 * np.pi * D_eff[i - 1] * freq * N_ar[i - 1] / field.c

        prop[0, 0] = np.exp(-1.0j * ptx)
        prop[1, 1] = np.exp(1.0j * ptx)

        TM = np.dot(prop, TM)

    return TM


def find_reflectance(freq_ar, n, ni, d):

    N_ar = [1, n + (ni * 1j), 1]  # Index of refraction [air -> sample -> air]
    D_ar = [1, d, 1]  # Thicknesses [air -> sample -> air]

    layers = len(N_ar)  # Number of layers

    # Convert arrays to numpy arrays
    N_ar = np.array(N_ar, dtype=complex)
    D_ar = np.array(D_ar)

    # Initialize reflection, transmission arrays
    R = np.zeros(len(freq_ar))
    T = np.zeros(len(freq_ar))

    # Find reflction as function of frequency
    for i in range(len(freq_ar)):
        freq_i = freq_ar[i]

        # Initialize the right traveling wave in the last layer as 1 with zero phase
        E_out = np.array((1.0 + 0.0j, 0.0 + 0.0j), dtype=complex)

        TM = get_transfer_matrix_ang_inc(N_ar, D_ar, freq_i)
        E_p, E_n = np.dot(TM, E_out)

        # Find reflection, transmission, and phase coefficients at this frequency
        R[i] = ((E_n / E_p) * (np.conjugate(E_n) / np.conjugate(E_p))).real
        T[i] = ((E_out[0] / E_p) * (E_out[0] / np.conjugate(E_p))).real

    return 10 * np.log10(R)
