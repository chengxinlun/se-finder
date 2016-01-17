import pickle
import numpy as np
from base import read_raw_data, get_total_sid_list, mask_points, extract_fit_part, save_fig
from fe_temp_observed import FeII_template_obs
from astropy.modeling import models
from scipy.integrate import quad

def read_fit_res(sid):
    data_file = open("Fe2/" + sid + "/Fe2.pkl", "rb")
    res = pickle.load(data_file)
    data_file.close()
    return res

def calc_flux(res):
    # Separate the parameter and construct integrating function
    cont_func = models.PowerLaw1D(res[21], res[22], res[23])
    fe2_func = FeII_template_obs(res[0], res[1], res[2], res[3], res[4], res[5]) - cont_func
    hbeta_func = models.Gaussian1D(res[6], res[7], res[8]) - cont_func
    o3_func = models.Gaussian1D(res[18], res[19], res[20]) - cont_func
    # Integrate to get flux
    fe2_flux = quad(Fe2_func, 4000.0, 5500.0)
    hbeta_flux = quad(hbeta_func, -np.inf, np.inf)
    o3_flux = quad(o3_func, -np.inf, np.inf)
    return [fe2_flux, hbeta_flux, o3_flux]

