from fe_temp_observed import FeII_template_obs
import numpy as np

class FeII_template_int:
    center_l1 = FeII_template_obs.center_l1
    center_n3 = FeII_template_obs.center_n3
    i_l1 = FeII_template_obs.i_l1
    i_n3 = FeII_template_obs.i_n3
    shift_l1 = 0.0
    shift_n3 = 0.0
    width_l1 = 2000.0
    width_n3 = 2000.0
    i_r_l1 = 1.0
    i_r_n3 = 1.0


    def __init__(self, shift_l1, width_l1, i_r_l1, shift_n3, width_n3, i_r_n3):
        self.shift_l1 = shift_l1
        self.shift_n3 = shift_n3
        self.width_l1 = width_l1
        self.width_n3 = width_n3
        self.i_r_l1 = i_r_l1
        self.i_r_n3 = i_r_n3

    def loz(self, x, amp, vel, center):
        width = np.sqrt(3 / 2) * center * vel / 299792.458
        res = amp * width * width / (width * width + (x - center) * (x - center))
        return res

    def func(self, x):
        res = 0
        for i in range(len(self.center_l1)):
            res = res + self.loz(x, self.i_r_l1 * self.i_l1[i], self.width_l1, self.center_l1[i] + self.shift_l1)
        for i in range(len(self.center_n3)):
            res = res + self.loz(x, self.i_r_n3 * self.i_n3[i], self.width_n3, self.center_n3[i] + self.shift_n3)
        return res


    
