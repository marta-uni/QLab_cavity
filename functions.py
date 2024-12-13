import numpy as np


def crop_to_min_max(piezo_voltage, *readings):
    '''Crops oscilloscope readings in order to include just one
     frequency sweep from the piezo.

     Piezo voltage has to be always the first input.

     Handles from 2 to 4 channels (+timestamps)'''

    if len(piezo_voltage) == 0:
        raise ValueError("Piezo voltage cannot be empty.")

    # Find indices of minimum and maximum values
    # Assuming there will be just one full sweep in the dataset
    min_index = np.argmin(piezo_voltage)
    max_index = np.argmax(piezo_voltage)

    # Ensure the possibility to analyse "backwards" sweeps too
    start_index = min(min_index, max_index)
    end_index = max(min_index, max_index)
    
    pv = piezo_voltage[start_index:end_index + 1]

    if not (2 <= len(readings) <= 4):
        raise ValueError("Invalid number of lists.")

    cropped_lists = [pv]
    for rd in readings:
        if not isinstance(rd, np.ndarray):
            print(type(rd))
            raise TypeError("All arguments must be np.arrays.")
        cropped = rd[start_index:end_index + 1]
        cropped_lists.append(cropped)

    return tuple(cropped_lists)


def fit_piezo_line(time, piezo_voltage):
    '''Converts timestamps in voltages on piezo.

    Returns voltages from a linear interpolation of input data.'''

    if len(time) == 0 or len(piezo_voltage) == 0 or len(time) != len(piezo_voltage):
        return None  # Return None if the input arrays are empty or of different lengths

    # Fit a line (degree 1 polynomial) to the piezo voltage data
    slope, intercept = np.polyfit(time, piezo_voltage, 1)
    piezo_fit = slope * time + intercept

    print('Fitting a*x+b:')
    print(f'slope = {slope} V/s\t intercept = {intercept} V')

    return piezo_fit
