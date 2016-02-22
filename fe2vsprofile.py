import pickle
import numpy as np
import matplotlib.pyplot as plt

filein = open('flux.pkl',  'rb')
result_dict = pickle.load(filein)
filein.close()

filein = open('profile-hbeta.pkl', 'rb')
profile_dict = pickle.load(filein)
filein.close()

hbeta = list()
fe2 = list()
prof = list()

fig = plt.figure()

for each in result_dict.keys():
    if result_dict[each][0]<0 or result_dict[each][1]<0 or result_dict[each][2]<0:
        continue
    if each not in profile_dict.keys():
        continue
    hbeta.append(result_dict[each][1])
    fe2.append(result_dict[each][0])
    prof.append(profile_dict[each])
    plt.annotate(s = str(each), xy = (profile_dict[each], result_dict[each][0] / result_dict[each][1]))
    print(each)

hbeta_a = np.array(hbeta)
fe2_a = np.array(fe2)

r_fe2 = fe2_a / hbeta_a

plt.scatter(prof, r_fe2)
plt.xlabel("FWHM / STD of Hbeta")
plt.ylabel("Relative flux of FeII")
plt.ylim(ymin  =  0.0, ymax = 1.1 * np.amax(r_fe2))
plt.axvline(2.355, 0, 1, label = "Gaussian", linestyle = 'dashed')
plt.show()
