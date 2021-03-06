import pickle
import numpy as np
from base import get_total_sid_list
from fe_temp_observed import FeII_template_obs
from astropy.modeling import models
import matplotlib.pyplot as plt


class FileNotFound(Exception):
    pass


# Read in the fit result
def read_fit_res(sid):
    try:
        data_file = open("Fe2/" + sid + "/Fe2.pkl", "rb")
    except Exception:
        raise FileNotFound
    res = pickle.load(data_file)
    data_file.close()
    return res


# Integrate to get flux
def calc_flux(res):
    # Separate the parameter and construct integrating function
    fe2_func = FeII_template_obs(res[0], res[1], res[2], res[3], res[4], res[5])
    # Integrate to get flux
    x = np.linspace(4000.0, 5500.0, 100000)
    fe2_flux = np.trapz(fe2_func(x), x)
    hbeta_flux = np.sqrt(2.0 * np.pi) * abs(res[8]) * res[6]
    o3_flux = np.sqrt(2.0 * np.pi) * abs(res[20]) * res[18]
    return [fe2_flux, hbeta_flux, o3_flux]


# Output calculation result
def output_flux(dic):
    fileout = open("flux.pkl", "wb")
    pickle.dump(dic, fileout)
    fileout.close()


def main_process(sid):
    res = read_fit_res(sid)
    [fe2, hbeta, o3] = calc_flux(res)
    return [fe2, hbeta, o3]


if __name__ == "main":
    dic = dict()
    sid_list = get_total_sid_list()
    for sid in sid_list:
        try:
            dic[str(sid)] = main_process(str(sid))
        except FileNotFound:
            print("Fit file not found: " + str(sid))
            continue
        print(dic[str(sid)])
        print("Process finished for " + str(sid))
    output_flux(dic)
