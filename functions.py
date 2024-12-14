import numpy as np
from scipy.signal import find_peaks, peak_widths
import matplotlib.pyplot as plt
from matplotlib import cm
from fit_peaks import lorentzian


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


def peaks(piezo_voltage, laser_voltage, height, distance, indices=False):
    '''
    Finds peaks in readings from the photodiode.

    Parameters:

    piezo_voltage: voltages on the piezo, should be the cleaned version (the output of fit_piezo_line)
    laser_voltage: voltages on the photodiode
    height: min height of the peaks
    distance: min number of point between a peak and the following one
    indices: bool to determine wheter to return peak indices as well

    Returns: 

    peaks_xvalues: voltage values on the piezo corresponding to detected peaks
    peaks: detected peaks
    scaled_widths: width of each peak in Volts
    indices (if bool indices == True): indices of the peaks
    '''
    peaks_indices, _ = find_peaks(
        laser_voltage, height=height, distance=distance)
    peaks = laser_voltage[peaks_indices]
    peaks_xvalues = piezo_voltage[peaks_indices]

    widths = peak_widths(laser_voltage, peaks_indices, rel_height=0.5)
    piezo_voltage_spacing = np.mean(np.diff(piezo_voltage))
    scaled_widths = widths[0]*piezo_voltage_spacing

    if indices:
        return peaks_xvalues, peaks, scaled_widths, peaks_indices
    else:
        return peaks_xvalues, peaks, scaled_widths


def scattering(x, y, x_label, y_label, title, file_name, save):
    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, label='Data', color='green')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()


def lin_quad_fits(x, y):
    '''
    Performs linear and quadratic fits and returns coefficients and errors.
    '''
    if (len(x) > 3):
        # Perform linear fit
        # coeffs = [a, b] for y = ax + b
        coeffs_1, V1 = np.polyfit(x, y, 1, cov=True)
        d_coeffs_1 = [np.sqrt(V1[0, 0]), np.sqrt(V1[1, 1])]
        # Perform quadratic fit
        # coeffs = [a, b, c] for y = ax^2 + bx + c
        coeffs_2, V2 = np.polyfit(x, y, 2, cov=True)
        d_coeffs_2 = [np.sqrt(V2[0, 0]), np.sqrt(V2[1, 1]), np.sqrt(V2[2, 2])]
        return coeffs_1, coeffs_2, d_coeffs_1, d_coeffs_2
    else:
        coeffs_1 = np.polyfit(x, y, 1)
        coeffs_2 = np.polyfit(x, y, 2)
        return coeffs_1, coeffs_2, [0, 0], [0, 0, 0]


def plot_fits(x, y, x_label, y_label, file_name, title=None, save=False):
    '''
    After performing linear and quadratic fits plots data and fit curves.

    Returns coefficients and errors for both fits.
    '''
    coeffs_1, coeffs_2, d_coeffs_1, d_coeffs_2 = lin_quad_fits(x, y)

    # Create polynomial functions from the coefficients
    linear_fit = np.poly1d(coeffs_1)
    quadratic_fit = np.poly1d(coeffs_2)

    # Generate x values for the fitted curve (same x range as the original data)
    x_fit = np.linspace(min(x), max(x), 100)  # Smooth line for plotting
    lin_fit = linear_fit(x_fit)  # Calculate the fitted line
    # Calculate corresponding y values for the fitted curve
    quad_fit = quadratic_fit(x_fit)

    # Format the coefficients for display
    lin_coeff_label = f"Linear Fit: y = ({coeffs_1[0]:.2e})x + ({coeffs_1[1]:.2e})"
    quad_coeff_label = (
        f"Quadratic Fit: y = ({coeffs_2[0]:.2e})x² + ({coeffs_2[1]:.2e})x + ({coeffs_2[2]:.2e})"
    )

    plt.figure(figsize=(12, 6))
    plt.scatter(x, y, label='Data', color='green', marker='x', s=30)
    plt.plot(x_fit, lin_fit, label=lin_coeff_label,
             color='blue', linestyle='--')  # Plot the linear fit
    plt.plot(x_fit, quad_fit, label=quad_coeff_label, color='red',
             linestyle='--')  # Plot the quadratic fit
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()
    return coeffs_1, coeffs_2, d_coeffs_1, d_coeffs_2


def plot_piezo_laser_fit(piezo_fitted, volt_laser, file_name, A, x0, gamma, xpeaks, ypeaks, width, save=False):
    fitted_curves = []
    for A_, x0_, gamma_, peak, w in zip(A, x0, gamma, xpeaks, width):
        x = np.linspace(peak - w * 1.2, peak + w * 1.2, 100)
        y = lorentzian(x, A_, x0_, gamma_)
        fitted_curves.append((x, y))

    cmap = cm.get_cmap('Oranges')
    colors = cmap(np.linspace(0.5, 0.9, len(fitted_curves)))

    plt.figure(figsize=(12, 6))
    plt.scatter(piezo_fitted, volt_laser, label='Laser Intensity vs. Piezo volt',
                color='green', marker='.')
    plt.scatter(xpeaks, ypeaks, marker='x', label='Peak Values')
    for i, (x, y) in enumerate(fitted_curves):
        plt.plot(x, y, '--', label=f'Fitted Lorentzian {i+1}', color=colors[i])
    plt.xlabel('Voltage Piezo (V)')
    plt.ylabel('Laser Intensity (V)')
    plt.title('Piezo Voltage vs Laser Voltage')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()
