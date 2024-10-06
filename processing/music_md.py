import numpy as np
from scipy.linalg import hankel
import numpy as np

def music_3d(domain_1, domain_2, domain_3, signal, n_sources, L, n_omegas=100, centralShift=False, show_plot=False, plot_zindex=0):

    def hankel2fold(signal, L):
        n_1, n_2 = signal.shape
        L_1, L_2 = L

        row_hankels = [hankel(signal[i][:L_2+1], signal[i][L_2:]) for i in range(n_1)]

        h = np.vstack(row_hankels[:L_1+1])

        for i in range(1, n_1 - L_1):
            h = np.hstack((h, np.vstack(row_hankels[i:i+L_1+1])))
        return h

    def hankel3fold(signal, L):
        n_0, n_1, n_2 = signal.shape
        L_0, L_1, L_2 = L

        row_hankels = [hankel2fold(signal[i], (L_1, L_2)) for i in range(n_0)]

        h = np.vstack(row_hankels[:L_0+1])

        for i in range(1, n_0 - L_0):
            h = np.hstack((h, np.vstack(row_hankels[i:i+L_0+1])))
        return h
    
    h = hankel3fold(signal, L)
    
    U, s, V = np.linalg.svd(h)
    Us = U[:, n_sources:]

    stepsize = domain_1[1] - domain_1[0]
    if centralShift:
        omega_grid_X, omega_grid_Y, omega_grid_Z = np.meshgrid(np.linspace(-1/(2*stepsize), 1/(2*stepsize), n_omegas), np.linspace(-1/(2*stepsize), 1/(2*stepsize), n_omegas), np.linspace(-1/(2*stepsize), 1/(2*stepsize), n_omegas))
    else:
        omega_grid_X, omega_grid_Y, omega_grid_Z = np.meshgrid(np.linspace(0, 1/stepsize, n_omegas), np.linspace(0, 1/stepsize, n_omegas), np.linspace(0, 1/stepsize, n_omegas))

    imaging_values = np.zeros((n_omegas, n_omegas, n_omegas))

    phi_x = domain_1[:L[0]+1]
    phi_y = domain_2[:L[1]+1]
    phi_z = domain_3[:L[2]+1]

    phi_X, phi_Y, phi_Z = np.meshgrid(phi_x, phi_y, phi_z)

    for i in range(n_omegas):
        for j in range(n_omegas):
            for k in range(n_omegas):
                omega_x = omega_grid_X[i][j][k]
                omega_y = omega_grid_Y[i][j][k]
                omega_z = omega_grid_Z[i][j][k]
                # Perform calculations for each point formed by omega_grid
                phi = np.exp(1j * (omega_x * phi_X + omega_y * phi_Y + omega_z * phi_Z))
                phi_vec = phi.reshape(-1)
                imaging_values[i][j][k] = np.linalg.norm(phi_vec) / np.linalg.norm(np.linalg.norm(np.matrix.getH(Us)@phi_vec))    
    
    if show_plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolormesh(omega_grid_X[plot_zindex], omega_grid_Y[plot_zindex], imaging_values[plot_zindex], shading='auto')
        plt.colorbar()
        plt.show()

    if centralShift:
        return np.linspace(-1/(2*stepsize), 1/(2*stepsize), n_omegas), imaging_values
    else:
        return np.linspace(0, 1/stepsize, n_omegas), imaging_values

    
        
if __name__ == '__main__':
    domain_1 = np.linspace(0, 1, 10)
    domain_2 = np.linspace(0, 1, 10)
    domain_3 = np.linspace(0, 1, 10)

    X, Y, Z = np.meshgrid(domain_1, domain_2, domain_3)

    signal = np.exp(1j * (2* Y))
    n_sources = 1
    L = (3, 3, 3)
    n_omegas = 100
    centralShift = False
    omegas, values = music_3d(domain_1, domain_2, domain_3, signal, n_sources, L, n_omegas)
    max_index = np.unravel_index(np.argmax(values), values.shape)
