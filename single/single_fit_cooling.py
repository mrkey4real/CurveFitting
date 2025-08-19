'''
====
COOLING
====
1min
====
Fitted coefficients for EIR:
 a = 0.26754729965852564, b = 0.042479035589165846, c = -0.0011410021860023254, d = -0.001979620032036456, e = 0.0005419221743030991, f = -0.00041962720373709246
Fitted coefficients for Capacity:
 a = 0.2990819007404593, b = 0.06373853327953119, c = -0.0010918736795737253, d = -0.018500011526982992, e = 0.00012837431450931483, f = 0.00037580720990527906
R-squared for EIR: 1.0
CVRMSE for EIR: 2.7907352655923925e-14%
R-squared for Capacity: 0.39624984281989883
CVRMSE for Capacity: 2.699871962559386%

====
5min
====
Fitted coefficients for EIR:
 a = 0.27119605676293895, b = 0.04215337325511754, c = -0.0011328911519477645, d = -0.0019698494568701416, e = 0.0005428165182418953, f = -0.00042201178219358906
Fitted coefficients for Capacity:
 a = -0.04134965654356429, b = 0.08554623999758465, c = -0.0013440688784973542, d = -0.012984669301598292, e = 0.00016394260117481607, f = 4.6339125795118975e-05
R-squared for EIR: 0.9999982238865069
CVRMSE for EIR: 0.012220088997841064%
R-squared for Capacity: 0.38539080031300144
CVRMSE for Capacity: 2.591708323135035%

====
15min
====
Fitted coefficients for EIR:
 a = 0.2761932711686852, b = 0.04150664485852297, c = -0.0011126434076651152, d = -0.0018033172570506553, e = 0.000543700449719611, f = -0.0004318693022460218
Fitted coefficients for Capacity:
 a = 0.17276845241387664, b = 0.07605067441555025, c = -0.0013137117463228867, d = -0.01984177505084244, e = 0.00017408003229580333, f = 0.0003143469708324316
R-squared for EIR: 0.9999927400391702
CVRMSE for EIR: 0.024635881025638663%
R-squared for Capacity: 0.3667688507846817
CVRMSE for Capacity: 2.1362564724274202%
'''
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

# Define the biquadratic function
def biquadratic(X, a, b, c, d, e, f):
    Twb, Tdb = X
    return a + b * Twb + c * Twb**2 + d * Tdb + e * Tdb**2 + f * Twb * Tdb

def cubic(Tdb, a, b, c, d):
    return a + b * Tdb + c * Tdb**2 + d * Tdb**3

def fit(data, ref_cap, ref_EIR):
    # Extract input variables
    Twb = data['Thermostat'].values
    Tdb = data['Air Temperature [degC]'].values
    
    # Extract output variables
    EIR = data['EIR'].values
    EIR = EIR/ref_EIR
    Capacity = data['Capacity_W'].values
    Capacity = Capacity/ref_cap
    
    # Perform curve fitting for Power
    # popt_EIR, pcov_EIR = curve_fit(cubic, Tdb, EIR)
    popt_EIR, pcov_EIR = curve_fit(biquadratic, (Twb,Tdb), EIR)
    a1, b1, c1, d1, e1, f1 = popt_EIR
    print(f"Fitted coefficients for EIR:\n a = {a1}, b = {b1}, c = {c1}, d = {d1}, e = {e1}, f = {f1}")
    
    popt_capacity, pcov_capacity = curve_fit(biquadratic, (Twb,Tdb), Capacity)
    a2, b2, c2, d2, e2, f2 = popt_capacity
    print(f"Fitted coefficients for Capacity:\n a = {a2}, b = {b2}, c = {c2}, d = {d2}, e = {e2}, f = {f2}")

    # Make predictions for both Power and Capacity
    predicted_EIR = biquadratic((Twb,Tdb), *popt_EIR)
    predicted_capacity = biquadratic((Twb,Tdb), *popt_capacity)
    # Calculate goodness of fit (R-squared) and CVRMSE for Power
    residuals_EIR = EIR - predicted_EIR
    ss_res_EIR = np.sum(residuals_EIR**2)
    ss_tot_EIR = np.sum((EIR - np.mean(EIR))**2)
    r_squared_EIR = 1 - (ss_res_EIR / ss_tot_EIR)
    rmse_EIR = np.sqrt(ss_res_EIR / len(EIR))
    cvrmse_EIR = (rmse_EIR / np.mean(EIR)) * 100
    print(f"R-squared for EIR: {r_squared_EIR}")
    print(f"CVRMSE for EIR: {cvrmse_EIR}%")
    
    # Calculate goodness of fit (R-squared) and CVRMSE for Capacity
    residuals_capacity = Capacity - predicted_capacity
    ss_res_capacity = np.sum(residuals_capacity**2)
    ss_tot_capacity = np.sum((Capacity - np.mean(Capacity))**2)
    r_squared_capacity = 1 - (ss_res_capacity / ss_tot_capacity)
    rmse_capacity = np.sqrt(ss_res_capacity / len(Capacity))
    cvrmse_capacity = (rmse_capacity / np.mean(Capacity)) * 100
    print(f"R-squared for Capacity: {r_squared_capacity}")
    print(f"CVRMSE for Capacity: {cvrmse_capacity}%")
    
    # Plotting scatter
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Subplot 1: Measured vs. Predicted EIR
    axs[0].scatter(EIR, predicted_EIR, color='blue', alpha=0.7, label='Data points')
    axs[0].plot([EIR.min(), EIR.max()], [EIR.min(), EIR.max()], 'r--', label='Ideal fit (y=x)')
    # # Calculate ±15% deviation bounds (filled) for EIR
    # lower_bound_EIR = EIR * 0.85
    # upper_bound_EIR = EIR * 1.15
    # axs[0].fill_between(EIR, lower_bound_EIR, upper_bound_EIR, color='gray', alpha=0.2, label='±15% Deviation')
    # Calculate ±15% deviation bounds for EIR
    axs[0].plot([EIR.min(), EIR.max()], [EIR.min() * 0.85, EIR.max() * 0.85], 'b--', label='±15% Deviation')
    axs[0].plot([EIR.min(), EIR.max()], [EIR.min() * 1.15, EIR.max() * 1.15], 'b--')
    axs[0].set_title('Measured vs. Predicted EIR')
    axs[0].set_xlabel('Measured EIR')
    axs[0].set_ylabel('Predicted EIR')
    axs[0].set_xlim(min(EIR.min(), predicted_EIR.min()),max(EIR.max(), predicted_EIR.max()))
    axs[0].set_ylim(min(EIR.min(), predicted_EIR.min()),max(EIR.max(), predicted_EIR.max()))
    axs[0].legend(loc='lower right')
    axs[0].grid()
    
    # Subplot 2: Measured vs. Predicted Capacity
    axs[1].scatter(Capacity, predicted_capacity, color='green', alpha=0.7, label='Data points')
    axs[1].plot([Capacity.min(), Capacity.max()], [Capacity.min(), Capacity.max()], 'r--', label='Ideal fit (y=x)')
    ##Calculate ±15% deviation bounds (filled) for Capacity
    #lower_bound_capacity = Capacity * 0.85
    #upper_bound_capacity = Capacity * 1.15
    #axs[1].fill_between(Capacity, lower_bound_capacity, upper_bound_capacity, color='gray', alpha=0.2, label='±15% Deviation')
    # Calculate ±15% deviation bounds for Capacity
    axs[1].plot([Capacity.min(), Capacity.max()], [Capacity.min() * 0.85, Capacity.max() * 0.85], 'g--', label='±15% Deviation')
    axs[1].plot([Capacity.min(), Capacity.max()], [Capacity.min() * 1.15, Capacity.max() * 1.15], 'g--')
    axs[1].set_title('Measured vs. Predicted Capacity')
    axs[1].set_xlabel('Measured Capacity')
    axs[1].set_ylabel('Predicted Capacity')
    axs[1].set_xlim(min(Capacity.min(), predicted_capacity.min()),max(Capacity.max(), predicted_capacity.max()))
    axs[1].set_ylim(min(Capacity.min(), predicted_capacity.min()),max(Capacity.max(), predicted_capacity.max()))
    axs[1].legend(loc='lower right')
    axs[1].grid()
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig("single_scatter_plot.png")
    plt.show()
    
    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    
    # Subplot 1: Measured vs. Predicted EIR
    x_points = np.arange(len(EIR))  # Number of data points
    axs[0].plot(x_points, EIR, label='Measured EIR', color='blue', linewidth=2)
    axs[0].plot(x_points, predicted_EIR, label='Predicted EIR', color='red', linestyle='--', linewidth=2)
    axs[0].set_title('Measured vs. Predicted EIR')
    axs[0].set_xlabel('Data Point Index')
    axs[0].set_ylabel('EIR')
    axs[0].legend(loc='lower right')
    axs[0].grid()
    
    # Subplot 2: Measured vs. Predicted Capacity
    axs[1].plot(x_points, Capacity, label='Measured Capacity', color='blue', linewidth=2)
    axs[1].plot(x_points, predicted_capacity, label='Predicted Capacity', color='red', linestyle='--', linewidth=2)
    axs[1].set_title('Measured vs. Predicted Capacity')
    axs[1].set_xlabel('Data Point Index')
    axs[1].set_ylabel('Capacity')
    axs[1].legend(loc='lower right')
    axs[1].grid()
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig("single_model_accuracy.png")
    plt.show()



#%%Load data
ref_cap = 8030.147323
ref_EIR = 0.280194112
# data = pd.read_csv("../data/single/single_data_for_fitting.csv")
data = pd.read_csv("../data/single/single_data_for_fitting_30min.csv")
fit(data, ref_cap, ref_EIR)