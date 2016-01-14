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
def template_fit(wave, flux, error, img_directory):
    os.chdir("../../")
    fig = plt.figure()
    plt.plot(wave, flux)
    hbeta_complex_fit_func = \
            fe_temp_observed.FeII_template_obs(6.2, 2000.0, 2.6, 6.2, 2000.0, 2.6) + \
            models.Gaussian1D(3.6, 4853.30, 40.0) + \
            models.Gaussian1D(2.0, 4346.40, 2.0) + \
            models.Gaussian1D(2.0, 4101.73, 2.0) + \
            models.Gaussian1D(5.0, 4960.0, 6.0) + \
            models.Gaussian1D(20.0, 5008.0, 6.0) + \
            models.PowerLaw1D(flux[0], wave[0], - np.log(flux[-1]/flux[0]) / np.log(wave[-1]/wave[0]), fixed = {"x_0": True})
    fitter = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            start = time.time()
            fit = fitter(hbeta_complex_fit_func, wave, flux, weights = error, maxiter = 3000)
            print("Time taken: ")
            print(time.time() - start)
        except SpectraException:
            expected = np.array(fit(wave))
            plt.plot(wave, expected)
            cont = models.PowerLaw1D(fit.parameters[21], fit.parameters[22], fit.parameters[23])
            plt.plot(wave, cont(wave))
            save_fig(fig, directory, "failed")
            plt.close()
            raise SpectraException("Fit failed")
    expected = np.array(fit(wave))
    cont = models.PowerLaw1D(fit.parameters[21], fit.parameters[22], fit.parameters[23])
    plt.plot(wave, expected)
    plt.plot(wave, cont(wave))
    save_fig(fig, directory, "succeed")
    plt.close()
    rcs = 0
    for i in range(len(flux)):
        rcs = rcs + (flux[i] - expected[i]) ** 2.0
    rcs = rcs / np.abs(len(flux)-17)
    if rcs > 10.0:
        raise SpectraException("Reduced chi-square too large: " + str(rcs))
    return fit.parameters, rcs


# Function to output fit result and error
def output_fit(fit_result, sid, line):
    picklefile = open(
        "Fe2/" +
        str(sid) +
        "/" +
        str(line) +
        ".pkl",
        "wb")
    pickle.dump(fit_result, picklefile)
    picklefile.close()


# Exception logging process
def exception_logging(sid, line, reason):
    log = open("Fe2_fit_error.log", "a")
    log.write(
        str(sid) +
        " " +
        str(line) +
        " " +
        str(reason) +
        "\n")
    log.close()


def main_process(sid, line):


line = [4000.0, 4902.0, 5500.0]
try:
    os.mkdir("Fe2")
except OSError:
    pass
try:
    os.mkdir("Fe2-fig")
except OSError:
    pass
sid_list = get_total_sid_list()
# sid_list = [1141]
for each_sid in sid_list:
    main_process(str(each_sid), line)
    #except Exception as reason:
    #    print(str(reason))
    #    pass
