import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

coefficients = pd.read_csv('sidebands_calib/calibration_coefficients.csv')

coeff1_df = coefficients.iloc[:4]
coeff2_df = coefficients.iloc[4:]

weights_1_slope = [1 / (u ** 2) for u in coeff1_df['d_slope']]
weights_2_slope = [1 / (u ** 2) for u in coeff2_df['d_slope']]
weights_1_intercept = [1 / (u ** 2) for u in coeff1_df['d_intercept']]
weights_2_intercept = [1 / (u ** 2) for u in coeff2_df['d_intercept']]

slope_1 = sum(x * w for x, w in zip(coeff1_df['slope'], weights_1_slope)) / sum(weights_1_slope)
slope_2 = sum(x * w for x, w in zip(coeff2_df['slope'], weights_2_slope)) / sum(weights_2_slope)
intercept_1 = sum(x * w for x, w in zip(coeff1_df['intercept'], weights_1_intercept)) / sum(weights_1_intercept)
intercept_2 = sum(x * w for x, w in zip(coeff2_df['intercept'], weights_2_intercept)) / sum(weights_2_intercept)

data_folder = 'data_non_confocal/clean_data'

def calibrate_write(data, slope, intercept, filepath):
    data['frequencies'] = slope * data['volt_piezo'] + intercept
    data.to_csv(filepath, index=False)

data6 = pd.read_csv(f'{data_folder}/error00006.csv')
calibrate_write(data6, slope_1, intercept_1, f'{data_folder}/calibrated_error00006_filtered.csv')
data7 = pd.read_csv(f'{data_folder}/error00007.csv')
calibrate_write(data6, slope_1, intercept_1, f'{data_folder}/calibrated_error00007_filtered.csv')
data16 = pd.read_csv(f'{data_folder}/error00016.csv')
calibrate_write(data6, slope_2, intercept_2, f'{data_folder}/calibrated_error00016_filtered.csv')