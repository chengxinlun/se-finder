import pickle
import numpy as np
import matplotlib.pyplot as plt

filein = open('flux.pkl', 'rb')
result_dict = pickle.load(filein)
filein.close()

hbeta = list()
fe2 = list()
o3 = list()

fig = plt.figure()

for each in result_dict.keys():
    if result_dict[each][0]<0 or result_dict[each][1]<0 or result_dict[each][2]<0:
        continue
    hbeta.append(result_dict[each][1])
    fe2.append(result_dict[each][0])
    o3.append(result_dict[each][2])
    plt.annotate(s = str(each), xy = (result_dict[each][2] / result_dict[each][1], result_dict[each][0] / result_dict[each][1]))

hbeta_a = np.array(hbeta)
fe2_a = np.array(fe2)
o3_a = np.array(o3)

r_fe2 = fe2_a / hbeta_a
r_o3 = o3_a / hbeta_a

plt.scatter(r_o3, r_fe2)
plt.show()
