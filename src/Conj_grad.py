import numpy as np
from scipy.signal import correlate
from scipy.signal import convolve
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_ghat(chopped_signal, b, num_iter, num_signals, rir, length_filter, init_guess = 'zeros', plot=False):

    # Variables specified by the user
    num_iter = num_iter  # Assuming we are only using N vectors for the conjugate basis. Also, it's the number of iterations
    Room_RIR = rir  # Plug in the measured RIR
    num_speaker = len(Room_RIR[0])
    J = num_speaker # number of speakers
    L = num_signals # number of signals ( focusing zones)
    M = length_filter  # Plug in a number for the length of our designed filter
    K = len(Room_RIR[0][0]) # RIR length
    delay_signal_length = int(len(b) / L) # length of each delayed signal
    N = delay_signal_length + 2 - M - K # length of the original signal z_p

    res_array = []

    # Conj gradient method
    # Here we have a system that is Ax = b where A is the convolution matrix derived from the RIR
    # In order to use the conj gradient method, we need it to be symmetric and positive definite
    # Therefore, we are actually solving for the system (A^T)Ax = (A^T)b

    # Obtaining the modified b = (A^*)b
    mod_b = A_adjoint_multiply(b, chopped_signal, Room_RIR, J, L, M, K, N)

    if init_guess == 'zeros':
        # Starting for the initial guess x_0 = [ 0 0 0 0 0 0 ......0]
        g_hat = np.zeros((M * L * J,))
    elif init_guess == 'ones':
        # Starting for the initial guess x_0 = [ uniform random numbers from 0 to 1]
        g_hat = np.ones((M * L * J,))
    elif init_guess == 'uni_random':
        # Starting for the initial guess x_0 = [ uniform random numbers from 0 to 1]
        g_hat = np.random.uniform(0, 1, (M * L * J,))

    A_multiply(g_hat,chopped_signal,Room_RIR,J,L,M,K,N)

    AAG = \
        A_adjoint_multiply(A_multiply(g_hat,chopped_signal,Room_RIR,J,L,M,K,N) ,chopped_signal, Room_RIR, J, L, M, K, N)

    res = mod_b - AAG
    P = res

    # # Now start the for loop to find the other vectors in the basis
    for n in tqdm(range(num_iter - 1)):
        alpha = np.inner(res, res) / np.inner(
            A_multiply(P, chopped_signal,Room_RIR,J,L,M,K,N), A_multiply(P, chopped_signal,Room_RIR,J,L,M,K,N))
        g_hat = g_hat + alpha * P
        res_prev = res
        res = res - alpha * A_adjoint_multiply(
            A_multiply(P,chopped_signal,Room_RIR,J,L,M,K,N) ,chopped_signal, Room_RIR, J, L, M, K, N)
        beta = np.inner(res, res) / np.inner(res_prev, res_prev)
        P = res + beta * P

        res_array.append(np.linalg.norm(res))

    if plot:
        plt.figure()
        plt.plot(mod_b)
        plt.title('A^*b')

        plt.figure()
        plt.title('Residual ')
        plt.plot(res_array)

        plt.figure()
        plt.plot(g_hat)
        plt.title('Estimated filter g_hat ')
        plt.show()


    return g_hat

def A_adjoint_multiply(x ,chopped_signal ,Room_RIR , J , L , M , K , N):

    block_signal_length = M + N + K -2

    result_mid = np.zeros((J * (M + N - 1 ),))
    result = np.zeros((M * L * J,))
    for j in range(J):
        for p in range(L):
            result_mid[j*(M+N-1):(j+1)*(M+N-1),] +=\
                correlate(x[p*block_signal_length:(p+1)*block_signal_length,], Room_RIR[p][j], mode='valid')
    count = 0
    for j in range(J):
        for p in range(L):
            result[count*M:(count+1)*M,] = \
                correlate(result_mid[j*(M+N-1):(j+1)*(M+N-1),], chopped_signal[p][:,j], mode='valid' )
            count +=1

    return result

def A_multiply(x ,chopped_signal ,Room_RIR , J , L , M , K , N, offset= 0 , num_ref = 0):

    block_signal_length = M + N + K -2

    result_mid = np.zeros((J * (M + N - 1 ),))
    for j in range(J):
        for p in range(L):
            result_mid[j*(M+N-1):(j+1)*(M+N-1),] +=\
                convolve(x[(j*L*M)+p*M:(j*L*M)+(p+1)*M,], chopped_signal[p][:,j])

    if offset != 0:
        result = np.zeros((num_ref * (block_signal_length),))

        for p in range(offset, offset + num_ref):
            for j in range(J):
                result[(p-offset)*(block_signal_length):(p-offset+1)*block_signal_length,] += \
                    convolve(result_mid[j*(M+N-1):(j+1)*(M+N-1),], Room_RIR[p][j])
    else:
        result = np.zeros((L * (block_signal_length),))

        for p in range(L):
            for j in range(J):
                result[(p-offset)*(block_signal_length):(p-offset+1)*block_signal_length,] += \
                    convolve(result_mid[j*(M+N-1):(j+1)*(M+N-1),], Room_RIR[p][j])

    return result
