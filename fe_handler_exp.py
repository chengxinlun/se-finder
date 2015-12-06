import math
import os
import numpy as np
import pickle
from scipy.integrate import quad
from scipy.stats import chisquare
import matplotlib.pylab as plt
from astropy.modeling import models, fitting
import warnings
import fe_temp_constructor
from base import read_raw_data, get_total_sid_list, mask_points, check_line, extract_fit_part

# Define a special class for raising any exception related during the fit


class SpectraException(Exception):
    pass


# Function to fit Hbeta for se quasars
def hbeta_complex_fit(wave, flux, error):
    fig = plt.figure()
    plt.plot(wave, flux)
    # (Hbeta, FeII, OIII, OIII, FeII)
    hbeta_complex_fit_func = models.Lorentz1D(5.0, 4853.0, 40.0, bounds = {"amplitude": [0, 50.0], "x_0": [4833,4873]}) + \
            models.Gaussian1D(5.0, 4930.0, 1.0, bounds = {"amplitude": [0, 20.0], "mean": [4900, 4950]}) + \
            models.Gaussian1D(3.0, 4959.0, 3.0, bounds = {"amplitude": [0, 25.0], "mean": [4950, 4970]}) + \
            models.Gaussian1D(5.0, 4961.0, 3.0, bounds = {"amplitude": [0, 25.0], "mean": [4955, 4970]}) + \
            models.Gaussian1D(20.0, 5007.0, 6.0, bounds = {"amplitude": [0, 50.0], "mean": [4990, 5020]}) + \
            models.Gaussian1D(5.0, 5018.0, 7.0, bounds = {"amplitude": [0, 25.0], "mean": [5013, 5030]}) + \
            models.Linear1D((flux[0] - flux[-1])/(wave[0]-wave[-1]), (-flux[0] * wave[-1] + flux[-1] * wave[0])/(wave[0]-wave[-1]))
    #hbeta_complex_fit_func.mean_3.tied = lambda x: -48.0 + x.mean_4
    #hbeta_complex_fit_func.amplitude_3.tied = lambda x: 1.0 / 2.99 * x.amplitude_4
    #hbeta_complex_fit_func.stddev_3.tied = lambda x: 1.0 * x.stddev_4
    fitter = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            fit = fitter(hbeta_complex_fit_func, wave, flux, weights = error, maxiter = 10000)
        except Warning:
            expected = np.array(fit(wave))
            plt.plot(wave, expected)
            cont = models.Linear1D(fit.parameters[18], fit.parameters[19])
            plt.plot(wave, cont(wave))
            fig.savefig("Hbeta-l-failed.jpg")
            plt.close()
            raise SpectraException("Line Hbeta fit failed")
    expected = np.array(fit(wave))
    plt.plot(wave, expected)
    cont = models.Linear1D(fit.parameters[18], fit.parameters[19])
    plt.plot(wave, cont(wave))
    fig.savefig("Hbeta-l.jpg")
    plt.close()
    rcs = 0
    for i in range(len(flux)):
        rcs = rcs + (flux[i] - expected[i]) ** 2.0
    rcs = rcs / np.abs(len(flux)-17)
    if rcs > 10.0:
        plt.close()
        raise SpectraException("Line Hbeta reduced chi-square too large" + str(rcs))
    return fit.parameters, rcs


# Function to fit Hbeta lines for non-se quasars
def hbeta_complex_fit_2(wave, flux, error):
    os.chdir("../../")
    fig = plt.figure()
    plt.plot(wave, flux)
    [cont_wave, cont_flux, cont_error] = extract_fit_part(wave, flux, error, 4040, 4060)
    [temp_wave, temp_flux, temp_error] = extract_fit_part(wave, flux, error, 5080, 5100)
    cont_wave = np.append(cont_wave, temp_wave)
    cont_flux = np.append(cont_flux, temp_flux)
    cont_error = np.append(cont_error, temp_error)
    cont_fitter = fitting.LevMarLSQFitter()
    cont = models.PowerLaw1D(cont_flux[0], cont_wave[0], - np.log(cont_flux[-1]/cont_flux[0]) / np.log(cont_wave[-1]/cont_wave[0]), fixed = {"x_0": True})
    cont_fit = cont_fitter(cont, cont_wave, cont_flux, weights = cont_error, maxiter = 10000)
    plt.plot(wave, cont_fit(wave))
    plt.show()
    flux = flux - cont_fit(wave)
    plt.plot(wave, flux)
    hbeta_complex_fit_func = \
            fe_temp_constructor.FeII_template(9900, 0, 1500.0, 4.0, 3.0, 3.0, 3.0, 1.0)  + \
            models.Gaussian1D(2.0, 4102.0, 5.0, bounds = {"amplitude": [0, 25.0], "mean": [4050, 4130]}) + \
            models.Gaussian1D(5.0, 4346.0, 20.0, bounds = {"amplitude": [0, 25.0], "mean": [4300, 4360]}) + \
            models.Gaussian1D(2.0, 4363.0, 0.5, bounds ={'amplitude': [0, 25.0], "mean": [4350, 4380]}) + \
            models.Gaussian1D(2.0, 4862.0, 1.0, bounds = {"amplitude": [0, 25.0], "mean": [4833, 4873], "stddev": [0.0, 7.0]}) + \
            models.Gaussian1D(5.0, 4860.0, 40.0, bounds = {"amplitude": [0, 25.0], "mean": [4833, 4873], "stddev": [7.0, 60.0]}) + \
            models.Gaussian1D(1.0, 4855.0, 70.0, bounds = {"amplitude": [0, 25.0], "mean": [4833, 5020]}) + \
            models.Gaussian1D(3.0, 4959.0, 12.0, bounds = {"amplitude": [0, 25.0], "mean": [4950, 4970]}) + \
            models.Gaussian1D(5.0, 4961.0, 3.0, bounds = {"amplitude": [0, 25.0], "mean": [4955, 4970]}) + \
            models.Gaussian1D(10.0, 5007.0, 3.0, bounds = {"amplitude": [0, 50.0], "mean": [4990, 5020]})
    #hbeta_complex_fit_func.mean_5.tied = lambda x: -48.0 + x.mean_6
    #hbeta_complex_fit_func.amplitude_5.tied = lambda x: 1.0 / 2.99 * x.amplitude_6
    #hbeta_complex_fit_func.stddev_5.tied = lambda x: 1.0 * x.stddev_6
    fitter = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            fit = fitter(hbeta_complex_fit_func, wave, flux, weights = error, maxiter = 30000)
        except SpectraException:
            expected = np.array(fit(wave))
            plt.plot(wave, expected)
            plt.show()
            fig.savefig("Hbeta-g-failed.jpg")
            plt.close()
            raise SpectraException("Line Hbeta fit failed")
    expected = np.array(fit(wave))
    plt.plot(wave, expected)
    plt.show()
    fig.savefig("Hbeta-g.jpg")
    plt.close()
    rcs = 0
    for i in range(len(flux)):
        rcs = rcs + (flux[i] - expected[i]) ** 2.0
    rcs = rcs / np.abs(len(flux)-17)
    if rcs > 10.0:
        plt.close()
        raise SpectraException("Line Hbeta reduced chi-square too large" + str(rcs))
    return fit.parameters, rcs


# Function to find union of ranges
def union(a):
    b = []
    for begin,end in sorted(a):
        if b and b[-1][1] >= begin - 1:
            b[-1] = (b[-1][0], end)
        else:
            b.append((begin, end))
    return b


# Add up flux
def flux_sum(part, wave, flux):
    sum_flux = 0
    for each in part:
        for i in range(len(wave)):
            if wave[i] < each[0]:
                continue
            if wave[i] > each[1]:
                break
            sum_flux = sum_flux + flux[i]
    return sum_flux


# Compare OIII and Fe2
def compare_fe2(wave, flux, error):
    # First fit Hbeta and OIII
    se = False
    [wave_fit, flux_fit, error_fit] = extract_fit_part(wave, flux, error, 4000.0, 5500.0)
    #try:
    #    [hbeta_g_res, g_rcs] = hbeta_complex_fit_2(wave_fit, flux_fit, error_fit)
    #except SpectraException:
    #    se = True
    #    g_rcs = 65535
    #try:
    [hbeta_l_res, l_rcs] = hbeta_complex_fit_2(wave_fit, flux_fit, error_fit)
    #except SpectraException:
    #    if se==True:
    #        raise SpectraException("Fit for Hbeta complex failed")
    #    else:
    #        l_rcs = 65535
    #        pass
    #if g_rcs>l_rcs:
    #    hbeta_res = hbeta_l_res
    #    o3range = [(hbeta_l_res[13] - 2.0 * hbeta_l_res[14], hbeta_l_res[13] + 2.0 * hbeta_l_res[14])]
    #    cont = lambda x: hbeta_l_res[18] * x + hbeta_l_res[19]
    #    hbetarange = [(hbeta_l_res[1] - 2.0 * hbeta_l_res[2], hbeta_l_res[1] + 2.0 * hbeta_l_res[2])]
    #else:
    #    hbeta_res = hbeta_g_res
    #    o3range = [(hbeta_g_res[19] - 2.0 * hbeta_g_res[20], hbeta_g_res[19] + 2.0 * hbeta_g_res[20])]
    #    cont = lambda x: hbeta_g_res[24] * x +hbeta_g_res[25]
    #    hbetarange = union([(hbeta_g_res[1] - 2.0 * hbeta_g_res[2], hbeta_g_res[1] + 2.0 * hbeta_g_res[2]), 
    #        (hbeta_g_res[4] - 2.0 * hbeta_g_res[5], hbeta_g_res[4] + 2.0 * hbeta_g_res[5])])
    #o3flux = flux_sum(o3range, wave, flux) - flux_sum(o3range, wave, list(map(cont, wave)))
    #hbetaflux = flux_sum(hbetarange, wave, flux) - flux_sum(hbetarange, wave, list(map(cont, wave)))
    #o3hb = o3flux / hbetaflux
    return [hbeta_l_res, 0, 0, 0, 0]


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
    os.chdir("Fe2")
    try:
        os.mkdir(str(sid))
    except OSError:
        pass
    os.chdir("../Fe2-fig")
    try:
        os.mkdir(str(sid))
    except OSError:
        pass
    os.chdir("../")
    [wave, flux, fluxerr] = read_raw_data(sid)
    [wave, flux, fluxerr] = mask_points(wave, flux, fluxerr)
    # Extract the part of data for fitting
    [wave, flux, fluxerr] = extract_fit_part(wave, flux, fluxerr, line[0], line[2])
    os.chdir("Fe2-fig/" + str(sid))
    #try:
    [hbeta, bef, aft, o3sn, fesn] = compare_fe2(wave, flux, fluxerr)
    #except Exception as reason:
    #    print(str(reason))
    #    exception_logging(sid, "Fe2", reason)
    #    os.chdir("../../")
    #    return
    os.chdir("../../")
    output_fit(hbeta, sid, "Hbeta")
    output_fit(bef, sid, "bef")
    output_fit(aft, sid, "aft")
    os.chdir("Fe2/")
    sn_file = open(str(sid) + ".txt", "a")
    sn_file.write("%9.4f    %9.4f\n" % (o3sn, fesn))
    sn_file.close()
    os.chdir("../")
    print(o3sn, fesn)
    print("Process finished for " + str(sid))


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
#sid_list = [521, 1039]
sid_list = [1141]
for each_sid in sid_list:
    #try:
    main_process(str(each_sid), line)
    #except Exception as reason:
    #    print(str(reason))
    #    pass
