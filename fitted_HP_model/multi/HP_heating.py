'''
1. Heating Normal Mode Curve-fitting Results:
Rated heating capacity [W]: 8440.446821
Rated heating COP [1]: 3.20
Rated air flow rate [m3/s]: 0.47194745

[Biquadratic curve:]
Fitted coefficients for Power:
 a = 13488.610646127694, b = -1272.8606538660847, c = 36.194375623543, d = 3402.405699942505, e = -0.46100233695388365, f = -160.73112719367995
Fitted coefficients for Capacity:
 a = 7883.831946513015, b = -1591.174492957941, c = 74.31613727343391, d = -6794.455548444492, e = -0.20752630677670952, f = 329.5821471281572
R-squared for Power: 0.6291369212534488
CVRMSE for Power: 4.519213305859266%
R-squared for Capacity: 0.9806717250359581
CVRMSE for Capacity: 4.083105972697881%

[Quadratic curve:]
Fitted coefficients for Power:
 a = 2748.183488438604, b = -1.7231106073628913e-05, c = -3.3830198685808695e-06, d = 9.194979389261801, e = -0.46100250694344097, f = -9.308443702206726e-05
Fitted coefficients for Capacity:
 a = 7413.513315016351, b = -1.1010662538786545e-14, c = -1.4046337545554842e-16, d = 163.39139395605747, e = -0.20752759650212282, f = -7.473041611565323e-05
R-squared for Power: 0.6291369212535389
CVRMSE for Power: 4.519213305858715%
R-squared for Capacity: 0.9806717250360322
CVRMSE for Capacity: 4.0831059726900625%


2. Heating Boost Mode Curve-fitting Results:
Rated heating capacity [W]: 8909.360533
Rated heating COP [1]: 3.04
Rated air flow rate [m3/s]: 0.47194745

[Biquadratic curve:]
Fitted coefficients for Power:
 a = 71088.19766872191, b = -4669.409371748502, c = 68.03832551173264, d = 6513.526803243286, e = -0.014793522536335434, f = -307.44811682471357
Fitted coefficients for Capacity:
 a = 25552.943957440904, b = -1870.1194687733826, c = 48.20884628884435, d = 6706.91577548721, e = 0.38133290946437476, f = -309.0039030987217
R-squared for Power: 0.9357053572055611
CVRMSE for Power: 2.693714752582025%
R-squared for Capacity: 0.9904488809869557
CVRMSE for Capacity: 3.0711573263202525%

[Quadratic curve:]
Fitted coefficients for Power:
 a = 2835.032452977851, b = -4.179247268303278e-14, c = -3.1405802955510703e-09, d = 22.957418352106362, e = -0.014801521961839346, f = -9.435433117075371e-05
Fitted coefficients for Capacity:
 a = 7558.314637294907, b = -1.2103831436749448e-14, c = -1.4342649312644782e-16, d = 183.50180581513848, e = 0.3813361794426207, f = -8.262697429464864e-05
R-squared for Power: 0.9357053572207374
CVRMSE for Power: 2.6937147522641083%
R-squared for Capacity: 0.9904488809869976
CVRMSE for Capacity: 3.071157326313534%

'''
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

# Define the biquadratic function
def biquadratic(X, a, b, c, d, e, f):
    Tindoor, Toutdoor = X
    return a + b * Tindoor + c * Tindoor**2 + d * Toutdoor + e * Toutdoor**2 + f * Tindoor * Toutdoor

# Load data
sheet_name = 'heating_normal'
#sheet_name = 'heating_boost'
data = pd.read_excel("heating_mode.xlsx", sheet_name=sheet_name)

# Extract input variables
Tindoor = data['iat_C'].values # indoor air dry-bulb temperature
Toutdoor = data['oat_C'].values # outdoor air dry-bulb temperature

# Extract output variables
Power = data['Power_W'].values
Capacity = data['Capacity_W'].values

# Define boundaries for Power and Capacity curves
# # biquadratic curve
# bounds_power = (
#     [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],  # Lower bounds
#     [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]  # Upper bounds
# )
# bounds_capacity = (
#     [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],  # Lower bounds
#     [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]  # Upper bounds
# )
# Fix b c and f as 0 to convert a biquadratic curve to a quadratic curve
bounds_power = (
    [-np.inf, -0.0001, -0.0001, -np.inf, -np.inf, -0.0001],  # Lower bounds
    [np.inf, 0, 0, np.inf, np.inf, 0]  # Upper bounds
)
bounds_capacity = (
    [-np.inf, -0.0001, -0.0001, -np.inf, -np.inf, -0.0001],  # Lower bounds
    [np.inf, 0, 0, np.inf, np.inf, 0]  # Upper bounds
)

# Perform curve fitting for Power
popt_power, pcov_power = curve_fit(biquadratic, (Tindoor, Toutdoor), Power, bounds=bounds_power)
a1, b1, c1, d1, e1, f1 = popt_power
print(f"Fitted coefficients for Power:\n a = {a1}, b = {b1}, c = {c1}, d = {d1}, e = {e1}, f = {f1}")

# Perform curve fitting for Capacity
popt_capacity, pcov_capacity = curve_fit(biquadratic, (Tindoor, Toutdoor), Capacity, bounds=bounds_capacity)
a2, b2, c2, d2, e2, f2 = popt_capacity
print(f"Fitted coefficients for Capacity:\n a = {a2}, b = {b2}, c = {c2}, d = {d2}, e = {e2}, f = {f2}")

# Make predictions for both Power and Capacity
predicted_power = biquadratic((Tindoor, Toutdoor), *popt_power)
predicted_capacity = biquadratic((Tindoor, Toutdoor), *popt_capacity)

# Calculate goodness of fit (R-squared) and CVRMSE for Power
residuals_power = Power - predicted_power
ss_res_power = np.sum(residuals_power**2)
ss_tot_power = np.sum((Power - np.mean(Power))**2)
r_squared_power = 1 - (ss_res_power / ss_tot_power)
rmse_power = np.sqrt(ss_res_power / len(Power))
cvrmse_power = (rmse_power / np.mean(Power)) * 100
print(f"R-squared for Power: {r_squared_power}")
print(f"CVRMSE for Power: {cvrmse_power}%")

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
# data['Predicted Power'] = predicted_power
# data['Predicted Capacity'] = predicted_capacity
# data.to_csv('fitted_results.csv', index=False)
# print("Fitted results saved to 'fitted_results.csv'")

# Plotting scatter
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Subplot 1: Measured vs. Predicted Power
axs[0].scatter(Power, predicted_power, color='blue', alpha=0.7, label='Data points')
axs[0].plot([Power.min(), Power.max()], [Power.min(), Power.max()], 'r--', label='Ideal fit (y=x)')
# # Calculate ±15% deviation bounds (filled) for Power
# lower_bound_power = Power * 0.85
# upper_bound_power = Power * 1.15
# axs[0].fill_between(Power, lower_bound_power, upper_bound_power, color='gray', alpha=0.2, label='±15% Deviation')
# Calculate ±15% deviation bounds for Power
axs[0].plot([Power.min(), Power.max()], [Power.min() * 0.85, Power.max() * 0.85], 'b--', label='±15% Deviation')
axs[0].plot([Power.min(), Power.max()], [Power.min() * 1.15, Power.max() * 1.15], 'b--')
axs[0].set_title('Measured vs. Predicted Power')
axs[0].set_xlabel('Measured Power (W)')
axs[0].set_ylabel('Predicted Power (W)')
axs[0].set_xlim(min(Power.min(), predicted_power.min()),max(Power.max(), predicted_power.max()))
axs[0].set_ylim(min(Power.min(), predicted_power.min()),max(Power.max(), predicted_power.max()))
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
axs[1].set_xlabel('Measured Capacity (W)')
axs[1].set_ylabel('Predicted Capacity (W)')
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

# Subplot 1: Measured vs. Predicted Power
x_points = np.arange(len(Power))  # Number of data points
axs[0].plot(x_points, Power, label='Measured Power', color='blue', linewidth=2)
axs[0].plot(x_points, predicted_power, label='Predicted Power', color='red', linestyle='--', linewidth=2)
axs[0].set_title('Measured vs. Predicted Power')
axs[0].set_xlabel('Data Point Index')
axs[0].set_ylabel('Power (W)')
axs[0].legend(loc='lower right')
axs[0].grid()

# Subplot 2: Measured vs. Predicted Capacity
axs[1].plot(x_points, Capacity, label='Measured Capacity', color='blue', linewidth=2)
axs[1].plot(x_points, predicted_capacity, label='Predicted Capacity', color='red', linestyle='--', linewidth=2)
axs[1].set_title('Measured vs. Predicted Capacity')
axs[1].set_xlabel('Data Point Index')
axs[1].set_ylabel('Capacity (W)')
axs[1].legend(loc='lower right')
axs[1].grid()

# Adjust layout and show plot
plt.tight_layout()
plt.savefig(sheet_name+"_model_accuracy.png")
plt.show()
