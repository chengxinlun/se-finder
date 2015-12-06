'''
Program to indentify se from a large amount of candidates roughly
'''

# Import necessary modules
import os
from base import read_raw_data, get_total_sid_list, extract_fit_part, mask_points
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling import models, fitting
import warnings


# Continuum fitting and subtraction
def cont_handler(wave, flux, ivar):
    [cont_wave, cont_flux, cont_error] = extract_fit_part(wave, flux, ivar, 4040, 4060)
    [temp_wave, temp_flux, temp_error] = extract_fit_part(wave, flux, ivar, 5080, 5100)
    cont_wave = np.append(cont_wave, temp_wave)
    cont_flux = np.append(cont_flux, temp_flux)
    cont_error = np.append(cont_error, temp_error)
    cont_fitter = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        cont = models.PowerLaw1D(cont_flux[0], cont_wave[0], - np.log(cont_flux[-1] / cont_flux[0]) / np.log(cont_wave[-1] / cont_wave[0]), fixed = {"x_0": True})
    cont_fit = cont_fitter(cont, cont_wave, cont_flux, weights = cont_error, maxiter = 10000)
    fig = plt.figure()
    plt.plot(wave, flux)
    plt.plot(wave, cont_fit(wave))
    flux_n = flux - cont_fit(wave)
    return [wave, flux_n, ivar, fig]


# Find max intensity in a range centered at wavelength with width width
def find_i(wave, flux, ivar, wavelength, width):
    [wave_i, flux_i, ivar_i] = extract_fit_part(wave, flux, ivar, wavelength - 0.5 * width, wavelength + 0.5 * width)
    i = max(flux_i)
    return i


# Process an object
def rough_finder(sid, hbeta, o3, fe2):
    [wave, flux, ivar] = read_raw_data(sid)
    [wave, flux, ivar] = extract_fit_part(wave, flux, ivar, 4000, 5500)
    [wave, flux, ivar] = mask_points(wave, flux, ivar)
    try:
        [wave, flux, ivar, fig] = cont_handler(wave, flux, ivar)
    except Exception:
        fig.savefig("rough_finder/" + str(sid))
        plt.close()
        return [-1, -1]
    fig.savefig("rough_finder/" + str(sid))
    plt.close() 
    try:
        i_hbeta = find_i(wave, flux, ivar, hbeta[0], hbeta[1])
        i_o3 = find_i(wave, flux, ivar, o3[0], o3[1])
    except ValueError:
        return [0, 0]
    i_fe2 = 0
    for each in fe2:
        try:
            i_fe2 = i_fe2 + find_i(wave, flux, ivar, each[0], each[1])
        except ValueError:
            continue
    o3vshbeta = i_o3 / i_hbeta
    fe2vshbeta = i_fe2 / i_hbeta
    return [o3vshbeta, fe2vshbeta]


# Plot o3 vs fe2
def plot_o3_vs_fe2(o3, fe2):
    fig = plt.figure()
    plt.scatter(o3, fe2)
    plt.xlabel("Relative Intensity of OIII 5007")
    plt.ylabel("Normalized Relative Intensity of Fe II")
    plt.show()
    fig.savefig("o3vsfe2.png")


try:
    os.mkdir("rough_finder")
except OSError:
    pass
hbeta = [4853.0, 20.0]
o3 = [5008.0, 6.0]
fe2 = [[4418.9, 4.0], [4449.6, 4.0], [4493.5, 4.0], [4522.6, 4.0], [4549.5, 4.0], [4583.8, 4.0], [5018.4, 4.0], [5169.0, 4.0], [5276.0, 4.0], [5316.0, 4.0]]
sid_list = get_total_sid_list()
#sid_list = [909]
i_o3 = list()
i_fe2 = list()
print(sid_list)
for each_sid in sid_list:
    print(each_sid)
    [temp1, temp2] = rough_finder(each_sid, hbeta, o3, fe2)
    print(temp1, temp2)
    if (temp1== -1 and temp2 == -1):
        print("Cannot fit continuum")
        continue
    if (temp1== 0 and temp2== 0):
        print("Cannot find Hbeta or OIII 5007")
        continue
    i_o3.append(temp1)
    i_fe2.append(temp2)
plot_o3_vs_fe2(i_o3, i_fe2)
