import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat, UFloat
from uncertainties.unumpy import nominal_values, std_devs, uarray
from scipy.odr import ODR, Model, RealData
import pandas as pd

def extract_nomserrs(array):
    noms = nominal_values(array)
    errs = std_devs(array)
    return noms, errs

def lin_model(params, x):
    m, b = params
    return m * x + b

def quad_model(params, x):
    a, b, c = params
    return a * x**2 + b * x + c

def plot_ufloat(x, y, x_label, y_label, title, file_name=None, save=False):
    """
    Plots data with uncertainties if data is given as ufloats and without uncertainties if data is a simple array.
    
    Parameters:
    - x, y: ufloat data or regular numerical data
    - x_label, y_label: Labels for x and y axes
    - title: Title of the plot
    - file_name: Name of the file to save the plot (optional)
    - save: Whether to save the plot (True) or show it (False)
    """
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    if all(isinstance(i, UFloat) for i in x) and all(isinstance(i, UFloat) for i in y):
        # Case 3: Both x and y are UFloat (plot with error bars in both directions)
        x_nom, x_err = extract_nomserrs(x)
        y_nom, y_err = extract_nomserrs(y)
        plt.errorbar(x_nom, y_nom, xerr=x_err, yerr=y_err, fmt='x', color='green', label='Data', capsize=3)

    elif all(isinstance(i, UFloat) for i in y) and not all(isinstance(i, UFloat) for i in x):
        # Case 2: Only y is UFloat (plot with error bars in y direction only)
        y_nom, y_err = extract_nomserrs(y)
        plt.errorbar(x, y_nom, yerr=y_err, fmt='x', color='green', label='Data', capsize=3)

    else:
        # Case 1: Neither x nor y are UFloat (plot without error bars)
        plt.plot(x, y, color='green', label='Data')


    # Plot formatting
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    
    # Save or show the plot
    if save:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()


def odr_fit(x, y, model_func, beta0):
    """Fit data using ODR with uncertainties in both x and y."""
    # Prepare the data depending on uncertainties
    if  all(isinstance(i, UFloat) for i in x) and all(isinstance(i, UFloat) for i in y):
        x_nom, x_err = extract_nomserrs(x)
        y_nom, y_err = extract_nomserrs(y)
        data = RealData(x_nom, y_nom, sx=x_err, sy=y_err)
    elif not all(isinstance(i, UFloat) for i in x) and all(isinstance(i, UFloat) for i in y):
        y_nom, y_err = extract_nomserrs(y)
        data = RealData(x, y_nom, sy=y_err)
    elif all(isinstance(i, UFloat) for i in x) and not all(isinstance(i, UFloat) for i in y):
        x_nom, x_err = extract_nomserrs(x)
        data = RealData(x_nom, y, sx=x_err)
    
    # Create the model
    model = Model(model_func)
    
    # Perform the ODR fit
    odr = ODR(data, model, beta0=beta0)
    output = odr.run()
    
    # Extract fitted parameters and their uncertainties
    params = output.beta
    param_errors = np.sqrt(np.diag(output.cov_beta))
    return params, param_errors

def fit_ufloat_data(x, y, model_func, beta0):
    """
    Fits data with uncertainties in both x and y directions using ODR. Handles cases
    where x or y may or may not have uncertainties (ufloats).

    Parameters:
    - x, y: ufloat data or regular numerical data
    - model_func: The fitting function model (e.g., linear_model or quadratic_model)
    - beta0: Initial guess for the parameters (optional)

    Returns:
    - fit_params: Fitted parameters with uncertainties
    - output: ODR result containing the fit details
    """
    # if either x or y has uncertainty perform odr fit, if they are both exact performe curvefit
    if  all(isinstance(i, UFloat) for i in x) or all(isinstance(i, UFloat) for i in y):
        fit_params, fit_param_errors = odr_fit(x, y, model_func, beta0)
    else:
        fit_params, fit_param_errors = curve_fit(model_func, x, y, p0=beta0)
        
    return fit_params, fit_param_errors

def plot_ufloat_fit(x, y, model_func, x_label, y_label, title, beta0, file_name, save):
    """
    Plots data with uncertainties in both x and y directions and the fitted curve.
    This function also fits the data using ODR and then plots the results.

    Parameters:
    - x, y: ufloat data
    - model_func: The fitting function model (e.g., lin_model)
    - x_label, y_label: Labels for x and y axes
    - title: Title of the plot
    - beta0: Initial guess for the parameters (optional)
    - file_name: Name of the file to save the plot (optional)
    - save: Whether to save the plot (True) or show it (False)
    """
    # Fit the data using the fit_ufloat_data function
    fit_params, fit_param_errs = fit_ufloat_data(x, y, model_func, beta0)

    # Extract data for plotting
    if all(isinstance(i, UFloat) for i in x):
        x_nom, x_err = extract_nomserrs(x)
    else:
        x_nom, x_err = x, None

    if all(isinstance(i, UFloat) for i in y):
        y_nom, y_err = extract_nomserrs(y)
    else:
        y_nom, y_err = y, None

    # Generate the fitted curve using nominal values of parameters
    x_fit = np.linspace(min(x_nom), max(x_nom), 100)
    y_fit = model_func(fit_params, x_fit)  # Pass nominal parameters to the model function

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot data with error bars
    plt.errorbar(x_nom, y_nom, xerr=x_err, yerr=y_err, fmt='x', color='green', label='Data', capsize=3)

    # Plot the fitted curve
    params = uarray(fit_params, fit_param_errs)
    fitstr = f'y = {params[0].nominal_value:.3g} ± {params[0].std_dev:.3g} * x + ' \
         f'{params[1].nominal_value:.3g} ± {params[1].std_dev:.3g}'
    plt.plot(x_fit, y_fit, label=fitstr, color='blue', linestyle='--')

    # Plot formatting
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # Save or show the plot
    if save:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()
    
    return fit_params, fit_param_errs

def calibrate(x, coeffs1):
    new_x = coeffs1[0] * x + coeffs1[1]
    return new_x

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_ufloat_fit_two(x, y, model_func1, model_func2, x_label, y_label, title, beta0_1, beta0_2, file_name, save):
    """
    Plots data with uncertainties in both x and y directions and the fitted curves for two models.
    This function also fits the data using ODR for both models and then plots the results.

    Parameters:
    - x, y: ufloat data
    - model_func1, model_func2: The two fitting function models (e.g., lin_model, quad_model)
    - x_label, y_label: Labels for x and y axes
    - title: Title of the plot
    - beta0_1, beta0_2: Initial guesses for the parameters for both models (optional)
    - file_name: Name of the file to save the plot (optional)
    - save: Whether to save the plot (True) or show it (False)
    """
    # Fit the data using the fit_ufloat_data function for both models
    fit_params1, fit_param_errs1 = fit_ufloat_data(x, y, model_func1, beta0_1)
    fit_params2, fit_param_errs2 = fit_ufloat_data(x, y, model_func2, beta0_2)

    # Extract data for plotting
    if all(isinstance(i, UFloat) for i in x):
        x_nom, x_err = extract_nomserrs(x)
    else:
        x_nom, x_err = x, None

    if all(isinstance(i, UFloat) for i in y):
        y_nom, y_err = extract_nomserrs(y)
    else:
        y_nom, y_err = y, None

    # Generate the fitted curves using nominal values of parameters
    x_fit = np.linspace(min(x_nom), max(x_nom), 100)
    y_fit1 = model_func1(fit_params1, x_fit)  # Pass nominal parameters to model_func1
    y_fit2 = model_func2(fit_params2, x_fit)  # Pass nominal parameters to model_func2

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot data with error bars
    plt.errorbar(x_nom, y_nom, xerr=x_err, yerr=y_err, fmt='x', color='green', label='Data', capsize=3)

    # Plot the fitted curves for both models
    params1 = uarray(fit_params1, fit_param_errs1)
    fitstr1 = f'Model 1: y = {params1[0].nominal_value:.3g} ± {params1[0].std_dev:.3g} * x + ' \
              f'{params1[1].nominal_value:.3g} ± {params1[1].std_dev:.3g}'
    plt.plot(x_fit, y_fit1, label=fitstr1, color='blue', linestyle='--')

    params2 = uarray(fit_params2, fit_param_errs2)
    fitstr2 = f'Model 2: y = {params2[0].nominal_value:.3g} ± {params2[0].std_dev:.3g} * x^2 + ' \
              f'{params2[1].nominal_value:.3g} ± {params2[1].std_dev:.3g} * x + ' \
              f'{params2[2].nominal_value:.3g} ± {params2[2].std_dev:.3g}'
    plt.plot(x_fit, y_fit2, label=fitstr2, color='red', linestyle='-.')

    # Plot formatting
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # Save or show the plot
    if save:
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()

    return fit_params1, fit_params2, fit_param_errs1, fit_param_errs2

def save_calibration_coefficients(calibration_coefficients, output_file):
    """
    Saves the calibration coefficients to a CSV file using pandas.

    Parameters:
    - calibration_coefficients: List of tuples containing file name, coefficients, and uncertainties.
    - output_file: Path to save the CSV file.
    """
    # Create a DataFrame from the calibration coefficients
    calibration_df = pd.DataFrame(
        calibration_coefficients,
        columns=["File Name", "slope", "d_slope", "intercept", "d_intercept"]
    )
    
    # Save to CSV
    calibration_df.to_csv(output_file, index=False)