import numpy as np
import pickle
from base import read_raw_data, get_total_sid_list, mask_points, extract_fit_part
from line_integration import read_fit_res

# Find nearest to index in flux and return its corespond in wave
def find_wave(wave, flux, index):
    return wave[(np.abs(flux - index)).argmin()]

# Calculate the full width at half maximum of a certain emission line
def calc_fwhm(wave, flux):
    up_flux = flux[0:flux.argmax()]
    up_wave = wave[0:flux.argmax()]
    down_flux = flux[flux.argmax(): -1]
    down_wave = wave[flux.argmax(): -1]
    wave_min = find_wave(up_wave, up_flux, 0.5 * np.amax(flux))
    wave_max = find_wave(down_wave, down_flux, 0.5 * np.amax(flux))
    return wave_max-wave_min

# Calculate the standard deviation of a certian emission line
def calc_std(wave, flux):
    ave = wave[flux.argmax()]
    std = np.sum((wave - ave) * (wave - ave) * flux) / np.sum(flux)
    return std


def main_process(sid):
    [wave, flux, error] = read_raw_data(str(sid))
    [wave, flux, error] = mask_points(wave, flux, error)
    fit_res = read_fit_res(sid)
    [wave, flux, error] = extract_fit_part(wave, flux, error, fit_res[7] - 2.0 * fit_res[8], fit_res[7] + 2.0 * fit_res[8])
    fwhm = calc_fwhm(wave, flux)
    std = fit_res[8]
    profile = fwhm / std
    print(sid + " " + str(profile))
    return profile


sid_list = get_total_sid_list()
res_dic = dict()
for sid in sid_list:
    try:
        res_dic[str(sid)] = main_process(str(sid))
    except Exception:
        continue
outfile = open("profile-hbeta.pkl", "wb")
pickle.dump(res_dic, outfile)
outfile.close()
