import pandas as pd
import functions as fn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from uncertainties import ufloat
import uncertainties.unumpy as unp

c = 3e8  # speed of light
R = 50e-3  # curvature radius

#####################################################################################################
# section with file specific values

title = 'bessel00000'
# find_peaks parameters
min_prominence = 0.01
min_distance = 900

path = f'data_imp_match/clean_data/{title}.csv'
figure_path = 'data_imp_match/figures/dips/'
os.makedirs(figure_path, exist_ok=True)

#####################################################################################################

'''read data'''
data = pd.read_csv(path)

reflection = data['reflection'].to_numpy()
volt_piezo = data['volt_piezo'].to_numpy()


'''find peaks, prominences and widths in data'''

dips_indices = find_peaks(-reflection, distance=min_distance,
                          prominence=min_prominence)[0]


dips_indices = np.delete(dips_indices, [-1])
dips_indices = np.delete(dips_indices, [0])

widths_ind = peak_widths(-reflection, dips_indices, rel_height=1)[0]

piezo_pk = volt_piezo[dips_indices]
refl_pk = reflection[dips_indices]


'''find linear fit of reflection'''

ranges_to_remove = [(int(dip - w // 2), int(dip + w // 2))
                    for dip, w in zip(dips_indices, widths_ind)]

remove_indices = set()
for start, end in ranges_to_remove:
    remove_indices.update(range(start, end))

refl_no_dips = [r for i, r in enumerate(reflection) if i not in remove_indices]
v_piezo_no_dips = np.array([v for i, v in enumerate(
    volt_piezo) if i not in remove_indices])

coeffs, V = np.polyfit(v_piezo_no_dips, refl_no_dips, 1, cov='unscaled')
x = np.linspace(volt_piezo[0], volt_piezo[-1], 100)
lin = coeffs[0] * x + coeffs[1]
slope = ufloat(coeffs[0], np.sqrt(V[0, 0]))
intercept = ufloat(coeffs[1], np.sqrt(V[1 ,1]))
lin_label = f'Slope = {slope}\nIntercept = {intercept} V'

'''find uncertainty on peak estimate'''

noise = refl_no_dips - (coeffs[0] * v_piezo_no_dips + coeffs[1])
pk_uncertainty = np.std(noise)
d_reflpk = [pk_uncertainty]*len(refl_pk)

'''plot'''

plt.figure()
plt.scatter(volt_piezo, reflection, label='Reflection',
            color='red', marker='.')
plt.errorbar(piezo_pk, refl_pk, yerr=d_reflpk, label='Dips', ls='',
             color='blue', marker='.', capsize=5)
plt.plot(x, lin, label=lin_label, color='green')
plt.xlabel('Volt piezo [V]')
plt.ylabel('Reflection signal [V]')
plt.yscale('log')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(figure_path + title + '_dips.png')


'''compute depths'''

refl_pk_unc = unp.uarray(refl_pk, d_reflpk)

baseline = ufloat(np.mean(refl_no_dips), pk_uncertainty)
prom_2 = baseline - refl_pk_unc

prom_2 = np.flip(prom_2, axis=0)
line_val = np.flip(slope * piezo_pk + intercept, axis=0)
refl_pk_unc = np.flip(refl_pk_unc, axis=0)
spacing = np.flip(np.diff(piezo_pk))


with open(f'data_imp_match/{title}.txt', 'w') as file:
    file.write(f'Sum of depths = {sum(prom_2)} V\n\n')
    file.write(lin_label)
    file.write('\n\nDepth of single dips in V:\n')
    file.write(str(prom_2))
    file.write('\n\n\"Baseline\" of single dips:\n')
    file.write(str(baseline))
    file.write('\n\nReflection values at dips:\n')
    file.write(str(refl_pk_unc))
    file.write('\n\nSpacings in piezo V:')
    file.write(str(spacing))

plt.show()
