import numpy as np
from fe_temp_int import FeII_template_int
from scipy.integrate import quad
import matplotlib.pyplot as plt

int_func_class = FeII_template_int(0.0, 2000.0, 1.0, 0.0, 2000.0, 1.0)
x = np.linspace(4000.0, 5500.0, 100000)
plt.plot(x, list(map(int_func_class.func, x)))
plt.show()
plt.close()
