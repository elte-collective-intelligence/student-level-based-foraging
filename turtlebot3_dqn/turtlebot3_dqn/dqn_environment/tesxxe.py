import numpy as np
from scipy.optimize import minimize

# Define the boundary conditions
theta_start = np.array([0, 0, 0])
theta_end = np.array([np.pi, np.pi/2, np.pi])

# Define the fifth-order polynomial coefficients
def coefficients(T):
    A = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0],
        [1, T, T**2, T**3, T**4, T**5],
        [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
        [0, 0, 2, 6*T, 12*T**2, 20*T**3]
    ])
    B = np.array([
        theta_start,
        [0, 0, 0],
        [0, 0, 0],
        theta_end,
        [0, 0, 0],
        [0, 0, 0]
    ]).flatten()
    return np.linalg.solve(A, B)

# Define the objective function to minimize T
def objective(T):
    coeffs = coefficients(T)
    max_velocity = np.max(np.abs(coeffs[1] + 2*coeffs[2]*T + 3*coeffs[3]*T**2 + 4*coeffs[4]*T**3 + 5*coeffs[5]*T**4))
    max_acceleration = np.max(np.abs(2*coeffs[2] + 6*coeffs[3]*T + 12*coeffs[4]*T**2 + 20*coeffs[5]*T**3))
    if max_velocity <= 1 and max_acceleration <= 0.25:
        return T
    else:
        return np.inf

# Find the minimum T
result = minimize(objective, x0=1, bounds=[(0.1, 10)])
T_min = result.x[0]

print(f"The minimum time duration T is {T_min:.2f} seconds")
