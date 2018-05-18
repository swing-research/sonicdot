import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, convolve
from scipy.fftpack import fft, ifft
import sounddevice as sd

def record_response(sine_sweep, speaker_num ,num_zeros, fs, n_mics, verbose = False):    
    sig_2_play = np.zeros(( len(sine_sweep) + num_zeros, speaker_num + 1))
    sig_2_play[:, speaker_num] = np.hstack( ( sine_sweep, np.zeros(num_zeros) ) )

    response = sd.playrec( sig_2_play , fs, channels=n_mics)
    sd.wait()    

    if verbose:
        #print('Delay of system ' delay_index)
        plt.figure()
        plt.plot(response)
        plt.show()
        
    return response

def calculate_rir(reponse, num_zeros, estimator, shorten = 0 ,verbose = False):
    #Get RIR #1
    padded_response = np.hstack( ( np.zeros(num_zeros), reponse, np.zeros( num_zeros ) ) )
    RIR = estimator.estimate( padded_response )
    
    if shorten != 0:
        RIR = RIR[:shorten]
    
    peek_x = np.argmax( RIR, axis=0 )
    if verbose:
        plt.plot(RIR)
        plt.show()
        
    return RIR


class RoomResponseEstimator(object):
    """
    Gives probe impulse, gets response and calculate impulse response.
    Method from paper: "Simultaneous measurement of impulse response and distortion with a swept-sine technique"
    Angelo Farina
    """

    def __init__(self, duration=10.0, low=100.0, high=15000.0, Fs=44100.0):
        self.Fs = Fs

        # Total length in samples
        self.T = Fs*duration
        self.w1 = low / self.Fs * 2*np.pi
        self.w2 = high / self.Fs * 2*np.pi

        self.probe_pulse = self.probe()

        # Apply exponential signal on the beginning and the end of the probe signal.
        exp_window = 1-np.exp(np.linspace(0,-10, 5000))
        self.probe_pulse[:exp_window.shape[0]] *= exp_window
        self.probe_pulse[-exp_window.shape[0]:] *= exp_window[-1::-1]

        # This is what the value of K will be at the end (in dB):
        kend = 10**((-6*np.log2(self.w2/self.w1))/20)
        # dB to rational number.
        k = np.log(kend)/self.T

        # Making reverse probe impulse so that convolution will just
        # calculate dot product. Weighting it with exponent to acheive 
        # 6 dB per octave amplitude decrease.
        self.reverse_pulse = self.probe_pulse[-1::-1] * \
            np.array(list(\
                map(lambda t: np.exp(t*k), np.arange(self.T))\
                )\
            )

        # Now we have to normilze energy of result of dot product.
        # This is "naive" method but it just works.
        Frp =  fft(fftconvolve(self.reverse_pulse, self.probe_pulse))
        self.reverse_pulse /= np.abs(Frp[round(Frp.shape[0]/4)])

    def probe(self):

        w1 = self.w1
        w2 = self.w2
        T = self.T

        # page 5
        def lin_freq(t):
            return w1*t + (w2-w1)/T * t*t / 2

        # page 6
        def log_freq(t):
            K = T * w1 / np.log(w2/w1)
            L = T / np.log(w2/w1)
            return K * (np.exp(t/L)-1.0)

        freqs = log_freq(range(int(T)))
        impulse = np.sin(freqs)
        return impulse

    def estimate(self, response):

        I = fftconvolve( response, self.reverse_pulse, mode='full')
        I = I[self.probe_pulse.shape[0]:self.probe_pulse.shape[0]*2+1]

        return I