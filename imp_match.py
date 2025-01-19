import pandas as pd
import functions as fn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.optimize import curve_fit

c = 3e8  # speed of light
R = 50e-3  # curvature radius

#####################################################################################################
# section with file specific values

title = 'imp_match00000'
path = f'data_imp_match/clean_data/{title}.csv'
figure_path = 'data_imp_match/figures/dips/'
os.makedirs(figure_path, exist_ok=True)

# find_peaks parameters
min_prominence = 0.035
min_distance = 200

#####################################################################################################

'''read data'''
data = pd.read_csv(path)

reflection = data['reflection'].to_numpy()
volt_piezo = data['volt_piezo'].to_numpy()


'''find peaks, prominences and widths in data'''

dips_indices = find_peaks(-reflection, distance=min_distance,
                          prominence=min_prominence)[0]
prominences_tuple = peak_prominences(-reflection, dips_indices)
prominences = prominences_tuple[0]

widths_ind = peak_widths(-reflection, dips_indices,
                         rel_height=1, prominence_data=prominences_tuple)[0]


'''find linear fit of reflection'''

ranges_to_remove = [(int(dip - w // 2), int(dip + w // 2))
                    for dip, w in zip(dips_indices, widths_ind)]

remove_indices = set()
for start, end in ranges_to_remove:
    remove_indices.update(range(start, end))

refl_no_dips = [r for i, r in enumerate(reflection) if i not in remove_indices]
v_piezo_no_dips = [v for i, v in enumerate(
    volt_piezo) if i not in remove_indices]

coeffs = np.polyfit(v_piezo_no_dips, refl_no_dips, 1)
x = np.linspace(volt_piezo[0], volt_piezo[-1], 100)
lin = coeffs[0] * x + coeffs[1]
lin_label = f'slope = {coeffs[0]:.2g}\nintercept = {coeffs[1]:.2g}'

piezo_pk = volt_piezo[dips_indices]
refl_pk = reflection[dips_indices]


'''compute prominence sum for 3 different cases'''

print('Without removing any peak')
print(f'Sum of depths = {sum(prominences)} V')

print('Consider clear even and odd modes')
mask = (dips_indices < 7000)
dips_indices = dips_indices[mask]
prominences = prominences[mask]
print(f'Sum of depths = {sum(prominences)} V')

print('Consider only even modes')
mask = (dips_indices > 2000) & (dips_indices < 7000)
dips_indices = dips_indices[mask]
prominences = prominences[mask]
print(f'Sum of depths = {sum(prominences)} V')


'''plot'''

plt.figure()
plt.scatter(volt_piezo, reflection, label='Reflection',
            color='red', marker='.')
plt.scatter(piezo_pk, refl_pk, label='Dips',
            color='blue', marker='x')
plt.plot(x, lin, label=lin_label, color='green')
plt.xlabel('Volt piezo [V]')
plt.ylabel('Reflection signal [V]')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(figure_path + title + '_dips.png')
plt.show()
