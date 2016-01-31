import numpy as np
from base import read_database, get_total_sid_list, mask_points, extract_fit_part

# The input of wave must in ascending order
def find_wave(wave, flux, index):
    for i in range(len(flux)):
        if flux[i]<index:
            continue
        if flux[i]>index:
            break
    return wave[i]


