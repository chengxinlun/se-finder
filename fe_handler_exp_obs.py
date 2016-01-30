import math
import os
import numpy as np
import pickle
from scipy.integrate import quad
from scipy.stats import chisquare
import matplotlib.pylab as plt
from astropy.modeling import models, fitting
import warnings
import fe_temp_observed
from base import read_raw_data, get_total_sid_list, mask_points, check_line, extract_fit_part, save_fig
import time

# Define a special class for raising any exception related during the fit


class SpectraException(Exception):
    pass


# Function to fit Hbeta lines for non-se quasars
def template_fit(wave, flux, error, img_directory, sid):
    # Fit continuum
    fig = plt.figure()
    plt.plot(wave, flux)
    [cont_wave, cont_flux, cont_error] = extract_fit_part(wave, flux, error, 4040, 4060)
    [temp_wave, temp_flux, temp_error] = extract_fit_part(wave, flux, error, 5080, 5100)
    cont_wave = np.append(cont_wave, temp_wave)
    cont_flux = np.append(cont_flux, temp_flux)
    cont_error = np.append(cont_error, temp_error)
    cont_fitter = fitting.LevMarLSQFitter()
    cont = models.PowerLaw1D(cont_flux[0], cont_wave[0], - np.log(cont_flux[-1]/cont_flux[0]) / np.log(cont_wave[-1]/cont_wave[0]), fixed = {"x_0": True})
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            cont_fit = cont_fitter(cont, cont_wave, cont_flux, weights = cont_error, maxiter = 10000)
        except Exception:
            save_fig(fig, img_directory, str(sid) + "-cont-failed")
            plt.close()
            raise SpectraException("Continuum fit failed")
    plt.plot(wave, cont_fit(wave))
    save_fig(fig, img_directory, str(sid) + "-cont-success")
    plt.close()
    # Fit emission lines
    flux = flux - cont_fit(wave)
    fig1 = plt.figure()
    plt.plot(wave, flux)
    hbeta_complex_fit_func = \
            fe_temp_observed.FeII_template_obs(6.2, 2000.0, 2.6, 6.2, 2000.0, 2.6) + \
            models.Gaussian1D(3.6, 4853.30, 40.0) + \
            models.Gaussian1D(2.0, 4346.40, 2.0) + \
            models.Gaussian1D(2.0, 4101.73, 2.0) + \
            models.Gaussian1D(5.0, 4960.0, 6.0) + \
            models.Gaussian1D(20.0, 5008.0, 6.0)
    fitter = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            start = time.time()
            fit = fitter(hbeta_complex_fit_func, wave, flux, weights = error, maxiter = 3000)
            print("Time taken: ")
            print(time.time() - start)
        except Exception:
            save_fig(fig1, img_directory, str(sid) + "-failed")
            plt.close()
            raise SpectraException("Fit failed")
    expected = np.array(fit(wave))
    plt.plot(wave, expected)
    save_fig(fig1, img_directory, str(sid) + "-succeed")
    plt.close()
    rcs = 0
    for i in range(len(flux)):
        rcs = rcs + (flux[i] - expected[i]) ** 2.0
    rcs = rcs / np.abs(len(flux)-17)
    if rcs > 10.0:
        raise SpectraException("Reduced chi-square too large: " + str(rcs))
    return fit.parameters, cont_fit.parameters, rcs


# Function to output fit result and error
def output_fit(fit_result, sid, band):
    picklefile = open("Fe2/" + str(sid) + "/" + band + ".pkl", "wb")
    pickle.dump(fit_result, picklefile)
    picklefile.close()


# Exception logging process
def exception_logging(sid, reason):
    log = open("Fe2_fit_error.log", "a")
    log.write(str(sid) + " " + str(reason) + "\n")
    log.close()


# Individual working process
def main_process(sid):
    print("Beginning process for " + str(sid))
    # Read data and preprocessing
    [wave, flux, error] = read_raw_data(sid)
    [wave, flux, error] = mask_points(wave, flux,  error)
    [wave, flux, error] = extract_fit_part(wave, flux, error, 4000.0, 5500.0)
    # Begin fitting and handling exception
    try:
        img_directory = "Fe2"
        [fit_res, cont_res, rcs] = template_fit(wave, flux, error,  img_directory, sid)
    except SpectraException as reason:
        exception_logging(sid, reason)
        print("Failed\n\n")
        return
    output_fit(fit_res, sid, "Fe2")
    output_fit(cont_res, sid, "cont")
    print("Finished\n\n")
        

# Getting total source list and setting up workspace directories
try:
    os.mkdir("Fe2")
except OSError:
    pass
sid_list = get_total_sid_list()
# sid_list = [1141]
os.chdir("Fe2")
for each_obj in sid_list:
    try:
        os.mkdir(str(each_obj))
    except OSError:
        pass
os.chdir("../")
# Start working process
for each_sid in sid_list:
    main_process(str(each_sid))
