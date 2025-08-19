'''
1. Cooling Mode Single-stage Curve-fitting Results:
Rated total cooling capacity [W]: 	8030.147323
Rated sensible heat ratio [1]	0.79
Rated cooling COP [1]: 	3.568954366
Rated air flow rate [m3/s]: 	0.471947443
Rated cooling EIR [1]:	0.280194112

Fitted coefficients for EIR:
 a = 0.20953728050672027, b = 0.03326870473245099, c = -0.0008936091618114271, d = -0.0015503881644627958, e = 0.0004244221139503266, f = -0.0003286435605208762
Fitted coefficients for Capacity:
 a = 1.4158175317037651, b = -0.049995224555762656, c = 0.0017585621335833943, d = 0.005601542076874012, e = -0.00020324020244063951, f = 8.339943831943936e-07
R-squared for EIR: 0.9996387003989011
CVRMSE for EIR: 0.4876682182822633%
R-squared for Capacity: 0.9968102120313624
CVRMSE for Capacity: 0.43059781418474574%

Fitted coefficients for EIR:
 a = 0.26754729948459177, b = 0.04247903556154983, c = -0.0011410021852605492, d = -0.001979620030749466, e = 0.0005419221739507913, f = -0.00041962720346429
Fitted coefficients for Capacity:
 a = 1.3848127617975963, b = -0.04890035664487451, c = 0.0017200535098787731, d = 0.0054789261260228175, e = -0.00019878967805449425, f = 8.131471251596344e-07
R-squared for EIR: 0.9996387003989016
CVRMSE for EIR: 0.487668218281947%
R-squared for Capacity: 0.9968102120322851
CVRMSE for Capacity: 0.43059781412246545%

Fitted coefficients for EIR:
 a = 0.26754729965852525, b = 0.04247903558916574, c = -0.0011410021860023213, d = -0.0019796200320364448, e = 0.0005419221743031003, f = -0.00041962720373709664
Fitted coefficients for Capacity:
 a = 1.5392492248392489, b = -0.089457139510662, c = 0.0024016296417590263, d = 0.010228079391473497, e = -0.00040877588550816923, f = 0.00046277054246848955
R-squared for EIR: 1.0
CVRMSE for EIR: 1.630110892686262e-13%
R-squared for Capacity: 0.5325907028722294
CVRMSE for Capacity: 2.975907387350975%

2. Heating Mode Single-stage Curve-fitting Results:
Rated heating capacity [W]: 	8323.218393
Rated heating COP [1]: 	3.7
Rated air flow rate [m3/s]:	0.472
Rated heating EIR [1]:	0.264321
Fitted coefficients for EIR:
 a = 0.9117153487018522, b = -0.019313029470144117, c = 0.0006996775026945997, d = -3.339988561271519e-05
Fitted coefficients for Capacity:
 a = 0.8217566524801142, b = 0.02448095547732557, c = 0.00020849098233205517, d = 5.729168852189226e-06
R-squared for EIR: 0.9890832906547032
CVRMSE for EIR: 3.602658443878954%
R-squared for Capacity: 0.997097948557767
CVRMSE for Capacity: 1.9470558796265378%


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
# Load data
# sheet_name = 'heating_boost'
# ref_cap = 7854.304681
# ref_EIR = 0.357765
ref_cap = 8030.147323
ref_EIR = 0.280194112
sheet_name = 'real_data'
# sheet_name = 'cooling_low'
data = pd.read_excel("cooling_mode.xlsx", sheet_name=sheet_name)

# Extract input variables
Twb = data['iat_C'].values # indoor air wet-bulb temperature
Tdb = data['oat_C'].values # outdoor air dry-bulb temperature

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
# a1, b1, c1, d1= popt_EIR
# print(f"Fitted coefficients for EIR:\n a = {a1}, b = {b1}, c = {c1}, d = {d1}")
# Perform curve fitting for Capacity
# popt_capacity, pcov_capacity = curve_fit(cubic, Tdb, Capacity)
popt_capacity, pcov_capacity = curve_fit(biquadratic, (Twb,Tdb), Capacity)
a2, b2, c2, d2, e2, f2 = popt_capacity
print(f"Fitted coefficients for Capacity:\n a = {a2}, b = {b2}, c = {c2}, d = {d2}, e = {e2}, f = {f2}")
# a2, b2, c2, d2 = popt_capacity
# print(f"Fitted coefficients for Capacity:\n a = {a2}, b = {b2}, c = {c2}, d = {d2}")
#%%
# Make predictions for both Power and Capacity
# predicted_EIR = cubic(Tdb, *popt_EIR)
# predicted_capacity = cubic(Tdb, *popt_capacity)
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

# # Save predictions to a CSV
# data['Predicted EIR'] = predicted_EIR
# data['Predicted Capacity'] = predicted_capacity
# data.to_csv('fitted_results.csv', index=False)
# print("Fitted results saved to 'fitted_results.csv'")

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
plt.savefig(sheet_name+"_scatter_plot.png")
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
plt.savefig(sheet_name+"_model_accuracy.png")
plt.show()
