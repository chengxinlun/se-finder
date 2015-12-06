'''
A place for frequently used funtions

Current list of functions for public use:
    read_raw_data: read the raw data with rmid = sid
    get_total_sid_list: get a list of all the rmid in the raw data base
    mask_points: delete points with error = 0
    check_line: check whether wavelength is inside the range of wave
    extract_fit_part: extract the part between min_wave and max_wave
'''

# Import necessary modules
import pickle
import os
import numpy as np


# Recommended method to read a single pkl file
def read_database(filein):
    readfile = open(filein, "rb")
    data = pickle.load(readfile)
    readfile.close()
    return data


# Recommended method to read raw data
def read_raw_data(sid):
    os.chdir("raw/"  +  str(sid))
    wave = read_database("wave.pkl")
    flux = read_database("flux.pkl")
    fluxerr = read_database("ivar.pkl")
    os.chdir("../../")
    return [wave, flux, fluxerr]


# Get the whole sid list
def get_total_sid_list():
    readfile = open("info_database/sid.pkl", "rb")
    sid_list = pickle.load(readfile)
    readfile.close()
    return sid_list


# Omit error=0 points
def mask_points(wave, flux, error):
    wave_temp = list()
    flux_temp = list()
    error_temp = list()
    for i in range(len(error)):
        if error[i] == 0:
            continue
        wave_temp.append(wave[i])
        flux_temp.append(flux[i])
        error_temp.append(error[i])
    return [wave_temp, flux_temp, error_temp]


# Check whether wavelength @wave_line is inside list of wave
def check_line(wave, wave_line):
    max_wave = max(wave)
    min_wave = min(wave)
    if ((max_wave > (wave_line  +  150.0)) and (min_wave < (wave_line  -  150.0))) == True:
        return True
    else:
        return False
                                    
                                    
# Function to extract part of the spectra to fit
def extract_fit_part(wave, flux, error, min_wave, max_wave):
    wave_fit = list()
    flux_fit = list()
    error_fit = list()
    for i in range(len(wave)):
        if wave[i] < min_wave:
            continue
        if wave[i] > max_wave:
            break
        wave_fit.append(wave[i])
        flux_fit.append(flux[i])
        error_fit.append(error[i])
    wave_fit = np.array(wave_fit)
    flux_fit = np.array(flux_fit)
    error_fit = np.array(error_fit)
    return [wave_fit, flux_fit, error_fit]
