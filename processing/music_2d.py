import numpy as np
from scipy.linalg import hankel
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import cvxpy as cp

def music_2d(domain_x, domain_y, signal, n_sources, L, n_omegas=100, show_plot=False, centralShift=False, freq_bound=None, show_peak=True, \
             weight_threshold=None, min_distance=0.2, returnWeights=False, unmodulated_signal=None):
    
    n_1, n_2 = signal.shape
    L_1, L_2 = L

    row_hankels = [hankel(signal[i][:L_2+1], signal[i][L_2:]) for i in range(n_1)]

    h = np.vstack(row_hankels[:L_1+1])

    for i in range(1, n_1 - L_1):
        h = np.hstack((h, np.vstack(row_hankels[i:i+L_1+1])))

    U, s, V = np.linalg.svd(h)
    Us = U[:, n_sources:]

    if freq_bound is not None:
        if centralShift:
            omega_grid_X, omega_grid_Y = np.meshgrid(np.linspace(-freq_bound[0], freq_bound[0], n_omegas), np.linspace(-freq_bound[1], freq_bound[1], n_omegas))
        else:
            omega_grid_X, omega_grid_Y = np.meshgrid(np.linspace(0, freq_bound[0], n_omegas), np.linspace(0, freq_bound[1], n_omegas))
    else:
        stepsize = domain_x[1] - domain_x[0]
        if centralShift:
            omega_grid_X, omega_grid_Y = np.meshgrid(np.linspace(-np.pi/(stepsize), np.pi/(stepsize), n_omegas), np.linspace(-np.pi/(stepsize), np.pi/(stepsize), n_omegas))
        else:
            omega_grid_X, omega_grid_Y = np.meshgrid(np.linspace(0, 2*np.pi/stepsize, n_omegas), np.linspace(0, 2*np.pi/stepsize, n_omegas))
    imaging_values = np.zeros((n_omegas, n_omegas))

    phi_x = domain_x[:L_1+1]
    phi_y = domain_y[:L_2+1]
    phi_X, phi_Y = np.meshgrid(phi_x, phi_y)

    for i in range(n_omegas):
        for j in range(n_omegas):
            omega_x = omega_grid_X[i][j]
            omega_y = omega_grid_Y[i][j]
            # Perform calculations for each point formed by omega_grid
            phi = np.exp(1j * (omega_x * phi_X + omega_y * phi_Y))
            phi_vec = phi.reshape(-1)
            imaging_values[i][j] = np.linalg.norm(phi_vec) / np.linalg.norm(np.linalg.norm(np.matrix.getH(Us)@phi_vec))

    # def imaging_func(omega_x, omega_y):
    #     phi = np.exp(1j * (omega_x * phi_X + omega_y * phi_Y))
    #     phi_vec = phi.reshape(-1)
    #     return np.linalg.norm(phi_vec) / np.linalg.norm(np.linalg.norm(np.matrix.getH(Us)@phi_vec))
    
    # imaging_values = imaging_func(omega_grid_X, omega_grid_Y)
    
    # Find local maxima

    # Define the neighborhood structure
    neighborhood = generate_binary_structure(2, 2)

    # Apply maximum filter to find local maxima
    local_maxima = maximum_filter(imaging_values, footprint=neighborhood) == imaging_values

    # # Find the n_sources largest local maxima
    # sorted_maxima = np.sort(imaging_values[local_maxima].flatten())[::-1]
    # if len(sorted_maxima) < n_sources:
    #     n_sources = len(sorted_maxima)
    # n_largest_maxima = sorted_maxima[:n_sources]

    # Exclude local maxima on the edge
    edge_mask = np.zeros_like(local_maxima)
    edge_mask[1:-1, 1:-1] = 1
    local_maxima = local_maxima & edge_mask
    
    # Find the n_sources largest local maxima
    sorted_maxima = np.sort(imaging_values[local_maxima].flatten())[::-1]
    if len(sorted_maxima) < n_sources:
        n_sources = len(sorted_maxima)
    n_largest_maxima = sorted_maxima[:n_sources]

    if show_plot:
        import matplotlib.pyplot as plt
        import scienceplots
        plt.style.use(['science', 'grid', 'ieee'])
        plt.pcolormesh(omega_grid_X, omega_grid_Y, imaging_values, shading='auto')
        plt.colorbar()
        
        # Plot local maxima
        # if show_peak:
        #     plt.scatter(omega_grid_X[local_maxima], omega_grid_Y[local_maxima], color='blue', marker='x', label='Local Maxima')

        # Plot the n_sources largest local maxima with different color
        results = []
        for i in range(n_sources):
            maxima_indices = np.where(imaging_values == n_largest_maxima[i])
            if show_peak:
                plt.scatter(omega_grid_X[maxima_indices], omega_grid_Y[maxima_indices], color='red', marker='x')
            results.append(np.squeeze([omega_grid_X[maxima_indices], omega_grid_Y[maxima_indices]]))
        plt.scatter([], [], c='red', marker='x', label='Local Maxima')
        plt.xlabel('$\mu_1$')
        plt.ylabel('$\mu_2$')
        plt.title('Imaging Values')
        plt.legend()
        plt.tight_layout()
        # plt.savefig('challenge/music_2d.pdf')
        plt.show()
        
    else:
        results = []
        for i in range(n_sources):
            maxima_indices = np.where(imaging_values == n_largest_maxima[i])
            results.append(np.squeeze([omega_grid_X[maxima_indices], omega_grid_Y[maxima_indices]]))

    if weight_threshold is not None:
        # print(results)
        weights = weight_solver(domain_x, domain_y, signal, results)
        results = [results[i] for i in range(n_sources) if weights[i] > weight_threshold]
    # Average arrays in results with distances smaller than min_distance
    if min_distance is not None:
        averaged_results = []
        for i in range(len(results)):
            current_result = results[i]
            if i == 0:
                averaged_results.append(current_result)
            else:
                distances = np.linalg.norm(current_result - np.array(averaged_results), axis=1)
                if np.min(distances) >= min_distance:
                    averaged_results.append(current_result)
                else:
                    closest_index = np.argmin(distances)
                    averaged_results[closest_index] = (averaged_results[closest_index] + current_result) / 2

        results = averaged_results
    
    if len(weights) >  len(results):
        weights = weight_solver(domain_x, domain_y, signal, results)
    if returnWeights:
        return results, weights
    else:
        return results
    
def weight_solver(domain_x, domain_y, signal, frequencies, modulate_term=np.eye(2)):

    k = len(frequencies)
    A = np.zeros((len(domain_x)*len(domain_y), k), dtype=np.complex128)
    domain_X, domain_Y = np.meshgrid(domain_x, domain_y)

    for i in range(k):
        omega_x, omega_y = frequencies[i]
        phi = np.exp(1j * (omega_x * domain_X + omega_y * domain_Y))
        A[:, i] = phi.reshape(-1)
    q =  - 2 * A.real.T @ signal.reshape(-1).real - 2 * A.imag.T @ signal.reshape(-1).imag

    # Define the quadratic optimization problem
    x = cp.Variable(k)
    P = A.real.T @ A.real + A.imag.T @ A.imag
    

    objective = cp.Minimize(cp.quad_form(x, P) + q @ x)
    prob = cp.Problem(objective, [0 <= x, cp.sum(x) == 1])

    prob.solve()

    return x.value

if __name__ == '__main__':
    
    x = np.linspace(0, 10, num=5)
    y = np.linspace(0, 10, num=5)
    X, Y = np.meshgrid(x, y)

    signal = 1/2 * np.exp(1j*(X*2 + Y*2)) + 2/3 * np.exp(1j*(0.5*X + 0.5*Y))

    imaging_domain, imaging_values = music_2d(x, y, signal, 2, (1, 1), n_omegas=100, show_plot=True, weight_threshold=0.2)

    

