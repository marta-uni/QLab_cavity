import pandas as pd
import functions as fn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, peak_prominences

c = 3e8  # speed of light
R = 50e-3  # curvature radius

#####################################################################################################
# section with file specific values

# if using the bessel file, sets all the correct parameters and removes unnecessary peaks the correct way
bessel = True

if bessel:
    title = 'bessel00000'
    # find_peaks parameters
    min_prominence = 0.01
    min_distance = 900
else:
    title = 'imp_match00000'
    # find_peaks parameters
    min_prominence = 0.035
    min_distance = 200

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

if bessel:
    dips_indices = np.delete(dips_indices, [-1])
    dips_indices = np.delete(dips_indices, [0])

prominences_tuple = peak_prominences(-reflection, dips_indices)
prominences = prominences_tuple[0]

widths_ind = peak_widths(-reflection, dips_indices,
                         rel_height=1, prominence_data=prominences_tuple)[0]

piezo_pk = volt_piezo[dips_indices]
refl_pk = reflection[dips_indices]


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


'''compute prominence sum'''

with open(f'data_imp_match/{title}.txt', 'w') as file:
    if bessel:
        file.write(f'Sum of depths = {sum(prominences)} V\n\n')
    else:
        file.write('Without removing any peak\n')
        file.write(f'Sum of depths = {sum(prominences)} V\n\n')

        file.write('Consider clear even and odd modes\n')
        mask = (dips_indices < 7000)
        dips_indices = dips_indices[mask]
        prominences = prominences[mask]
        file.write(f'Sum of depths = {sum(prominences)} V\n\n')

        file.write('Consider only even modes\n')
        mask = (dips_indices > 2000) & (dips_indices < 7000)
        dips_indices = dips_indices[mask]
        prominences = prominences[mask]
        file.write(f'Sum of depths = {sum(prominences)} V\n\n')

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


'''another way to compute prominences'''

prom_2 = coeffs[0] * piezo_pk + coeffs[1] - refl_pk

if bessel:
    with open(f'data_imp_match/{title}.txt', 'a') as file:
        file.write('Alternative calculation of prominence sum:\n\n')
        file.write(f'Sum of depths = {sum(prom_2)} V')

prominences = np.flip(prominences, axis=0)
prom_2 = np.flip(prom_2, axis=0)
line_val = np.flip(coeffs[0] * piezo_pk + coeffs[1], axis=0)
refl_pk = np.flip(refl_pk, axis=0)
spacing = np.flip(np.diff(piezo_pk))
end_peaks = refl_pk + prominences


print('prominences')
print(prominences)
print('line - depth')
print(prom_2)
print('line values')
print(line_val)
print('line coefficients')
print(coeffs)
print('depth values')
print(refl_pk)
print('end of peaks')
print(end_peaks)
print('spacing')
print(spacing)


bin_centers = range(0, len(prominences))

plt.figure()
plt.bar(bin_centers, prom_2, width=1, edgecolor='black', alpha=0.7)
plt.xlabel('Peak index')
plt.ylabel('Reflection dip [V]')
plt.grid()
plt.tight_layout()
plt.savefig(figure_path + title + '_histogram.png')
plt.show()
