import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data_confocal/clean_data/scope_15_calib.csv')
peaks = pd.read_csv('data_confocal/clean_data/scope_15_peaks.csv')

peaks_non_confoc = peaks['freq_non_confoc'].to_numpy()


plt.figure(figsize=(12, 6))
plt.scatter(data['freq_non_confoc'], data['volt_laser'], label='Data', color='green', s=8)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Laser Intensity [V]}')
plt.axvline((peaks_non_confoc[2] + peaks_non_confoc[0])/2, color='black', ls='--')
plt.axvline((peaks_non_confoc[4] + peaks_non_confoc[2])/2, color='black', ls='--')
plt.title('Frequency vs Laser Voltage')
plt.xlim(2.5e9, 6.5e9)
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data_confocal/figures/justaplot_1.png')
plt.close()