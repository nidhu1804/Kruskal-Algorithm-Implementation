import numpy as np
from scipy.spatial.distance import pdist, squareform

def euc_distances(conf):
    return squareform(pdist(conf, 'euclidean'))

def sym_matrix(values, size):
    upper = np.zeros((size, size))
    upper[np.triu_indices(size, 1)] = values
    upper = upper.T + upper
    np.fill_diagonal(upper, 0)
    return upper
def upper_matrix(values, size):
    upper = np.zeros((size, size))
    upper[np.triu_indices(size, 1)] = values
    return(upper)

def read_input_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    max_iterations = int(lines[0].strip())
    lambda_t = float(lines[2].strip())
    max_stress = float(lines[4].strip())

    input_lines = lines[6:10]
    configurations = []
    for line in input_lines:
        parts = line.strip().split(';')
        configurations.append([float(part) for part in parts])

    print("Initial Configurations:")
    print(configurations)

    D = [] #distance_matrix
    for line in lines[11:len(lines)-1]:
        row = list(map(float, line.strip().split(';')))
        D.extend(row)
    D = np.array(D)
    print("Distance Matrix D:\n", D)

    return max_iterations, lambda_t, max_stress, np.array(configurations), np.array(D)

def compute_stress(configurations,distance_matrix):
    DH=euc_distances(configurations)
    print("Euclidean Distances:\n", DH)
    D_upper = upper_matrix(distance_matrix,4)
    flat_D=D_upper.flatten()
    flat_DH=DH.flatten()
    indices=[i for i in np.argsort(flat_D) if flat_D[i] > 0]
    flat_DH[indices]
    d_hat = flat_DH[indices]
    d=flat_D[indices]
    print("indices:", indices)
    print("d_hat values:\n",d_hat)
    print("d:\n", d)
    delta = []
    for i in range(len(d_hat)):
        if i == 0:
            delta.append(d_hat[i])
        elif delta[-1] <= d_hat[i]:
            delta.append(d_hat[i])
        else:
            mean = (d_hat[i] + d_hat[i - 1]) / 2
            delta[-1] = mean
            delta.append(mean)
    print("delta values:", delta)
    n=D_upper.shape[0]
    d_bar = (2 / (n * (n - 1))) * np.sum(d_hat)
    print("d_bar value:", d_bar)
    b = np.sum(np.square(d_hat - delta))
    b_max = np.sum(np.square(d_hat - d_bar))
    stress=b/b_max
    print("stress value:", stress)
    return stress, d , d_hat, delta, flat_D, indices, D_upper

def update_configuration(lambda_t,configurations, d, d_hat, delta, flat_D, indices, D_upper):
    sym_d = sym_matrix(d, 4)
    # Initialize arrays for symmetric matrices of d_hat and delta
    sym_d_hat = np.zeros((len(flat_D)))
    sym_delta = np.zeros((len(flat_D)))
    # Assign values of d_hat and delta to the upper triangular part of their respective arrays
    sym_d_hat[indices] = d_hat
    sym_delta[indices] = delta
    #reshape arrays to matrix
    sym_d_hat = sym_d_hat.reshape((D_upper.shape))
    sym_delta = sym_delta.reshape((D_upper.shape))
    # Ensure symmetry of matrices by adding them to their transposes
    sym_d_hat = sym_d_hat + sym_d_hat.T
    sym_delta = sym_delta + sym_delta.T
    # Initialize gradient matrix
    grad_b = np.zeros_like(configurations)

    # Compute gradient for each configuration parameter
    for i in range(configurations.shape[0]):
        for k in range(configurations.shape[1]):
            for j in range(sym_d_hat.shape[1]):
                if j != i:
                    grad_b[i, k] += 2 * (sym_d_hat[i, j] - sym_delta[i, j]) * \
                                    (configurations[i, k] - configurations[j, k]) / sym_d_hat[i, j]

    # Calculate the new configuration
    new_config = configurations - lambda_t * grad_b
    print(new_config)
    return new_config

def kruskals_approach(input_file, output_file):
    max_iterations, lambda_t, max_stress, configurations, distance_matrix = read_input_file(input_file)
    with open(output_file, 'w') as file:
        for config in configurations:
            file.write(';'.join(map(str, config)) + '\n')
        file.write('====\n')
        initial_stress, d, d_hat, delta, flat_D, indices, D_upper= compute_stress(configurations, distance_matrix)
        file.write(f'{initial_stress:.3f}\n')
        file.write('====\n')

        for iteration in range(1, max_iterations + 1):
            config = update_configuration(lambda_t,configurations,d,d_hat,delta,flat_D, indices, D_upper)
            stress, d, d_hat, delta, flat_D, indices, D_upper= compute_stress(config, distance_matrix)
            file.write(f'{iteration:.3f}\n')
            file.write('====\n')
            for con in config:
                file.write(';'.join(map(str, con)) + '\n')
            file.write('====\n')
            file.write(f'{stress:.3f}\n')
            file.write('====\n')

            if stress <= max_stress:
                break

# Example usage
kruskals_approach('input.txt', 'output.txt')
