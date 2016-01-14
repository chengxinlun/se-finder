import fe_temp_observed as fe
import numpy as np
import matplotlib.pyplot as plt

y = fe.FeII_template_obs(0.0, 1980.0, 10.0, 0.0, 1980.0, 10.0)
x = np.linspace(4000.0, 5500.0, num = 5000)
plt.plot(x, y(x))
plt.show()
