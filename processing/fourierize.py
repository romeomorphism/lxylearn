import numpy as np
def empirical_cf_2d(samples, domain_x, domain_y, modulate_term=None, returnUnmodulated=False):
    
    domain_X, domain_Y = np.meshgrid(domain_x, domain_y)
    
    emp_cf = np.zeros(domain_X.shape, dtype=complex)
    emp_cf_unmodulated = np.zeros(domain_X.shape, dtype=complex)
    

    
    for i in range(emp_cf.shape[0]):
        for j in range(emp_cf.shape[1]):
            if modulate_term is not None:
                array = np.array([domain_X[i][j], domain_Y[i][j]])
                term = array @ modulate_term @ array.T / 2
                fourier_data = np.exp(1j * (domain_X[i][j] * samples[:, 0] + domain_Y[i][j] * samples[:, 1]) + term)
                emp_cf_unmodulated[i][j] = np.mean(np.exp(1j * (domain_X[i][j] * samples[:, 0] + domain_Y[i][j] * samples[:, 1])))
            else:
                fourier_data = np.exp(1j * (domain_X[i][j] * samples[:, 0] + domain_Y[i][j] * samples[:, 1]))
            emp_cf[i][j] = np.mean(fourier_data)
    
    if returnUnmodulated:
        return emp_cf, emp_cf_unmodulated
    else:
        return emp_cf
    
def empirical_cf_3d(samples, domain_1, domain_2, domain_3, modulate_term=None, returnUnmodulated=False):

    domain_X, domain_Y, domain_Z = np.meshgrid(domain_1, domain_2, domain_3)

    emp_cf = np.zeros(domain_X.shape, dtype=complex)
    emp_cf_unmodulated = np.zeros(domain_X.shape, dtype=complex)


    for i in range(emp_cf.shape[0]):
        for j in range(emp_cf.shape[1]):
            for k in range(emp_cf.shape[2]):
                if modulate_term is not None:
                    array = np.array([domain_X[i][j][k], domain_Y[i][j][k], domain_Z[i][j][k]])
                    term = array @ modulate_term @ array.T / 2
                    fourier_data = np.exp(1j * (domain_X[i][j][k] * samples[:, 0] + domain_Y[i][j][k] * samples[:, 1] + domain_Z[i][j][k] * samples[:, 2]) + term)
                    emp_cf_unmodulated[i][j][k] = np.mean(np.exp(1j * (domain_X[i][j][k] * samples[:, 0] + domain_Y[i][j][k] * samples[:, 1] + domain_Z[i][j][k] * samples[:, 2])))
                else:
                    fourier_data = np.exp(1j * (domain_X[i][j][k] * samples[:, 0] + domain_Y[i][j][k] * samples[:, 1] + domain_Z[i][j][k] * samples[:, 2]))
                emp_cf[i][j][k] = np.mean(fourier_data)
    
    if returnUnmodulated:
        return emp_cf, emp_cf_unmodulated
    else:
        return emp_cf

