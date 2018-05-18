import numpy as np
import random
import sounddevice as sd
import soundfile as sf
import scipy as sp

def read_wav_file(filename):
    '''
    Reads in a .wav file and returns the data within the .wav file and its
    sampling frequency.

    Arguments:
        filename (str): string of .wav file to read
    Returns: data, fs
        numpy.ndarray of data
        int sampling frequency
    '''
    return sf.read(filename, dtype='float32')

def cosShift(num_samples_shifting, max_samples_shifting):
    '''
    Takes in number of samples speakers have been shifting and the desired max
    number of samples to shift. Returns two weights that sum to one.

    Arguments:
        num_samples_shifting (int): number of samples since shift started
        max_samples_shifting (int): total number of samples to shift
    Returns: coef1, coef2
        numpy.float64 corresponding to current speaker weight
        numpy.float64 corresponding to nextspeaker weight
    '''
    cos_value = np.pi * (num_samples_shifting / max_samples_shifting)
    cur_speaker_weight = ( np.cos( cos_value ) + 1 ) / 2
    next_speaker_weight = 1 - cur_speaker_weight
    return cur_speaker_weight, next_speaker_weight

def create_shifted_sample(num_speakers, speaker_offset, num_samples_shifting, samples_to_use_shifting, active_speaker, next_speaker, sample):
    '''
    Takes in all information about sampel and returns an n speaker shifted version.

    Arguments:
        num_speakers            (int): number of speakes in array
        speaker_offset          (int): number of channels to offset
        num_samples_shifting    (int): number of samples been shifting for
        samples_to_use_shifting (int): number of samples a shift takes
        active_speaker          (int): speaker that has signal
        next_speaker            (int): the next speaker to have signal
        sample                  (float): the actual data

    Returns:
        array of values
    '''
    sample_data = [ 0 for i in range( num_speakers + speaker_offset ) ]
    mask = [ 0 for i in range( num_speakers + speaker_offset ) ]
    cur_speaker_weight ,next_speaker_weight = cosShift(num_samples_shifting, samples_to_use_shifting)
    sample_data[ active_speaker + speaker_offset ] = sample * cur_speaker_weight
    sample_data[ next_speaker + speaker_offset] = sample * next_speaker_weight
    mask[active_speaker + speaker_offset] = 1.0 * cur_speaker_weight
    mask[next_speaker + speaker_offset] = 1.0 * next_speaker_weight
    return sample_data , mask

def create_regular_sample(num_speakers, speaker_offset, active_speaker, sample):
    '''
    Takes in all information about sampel and returns an n speaker  version.

    Arguments:
        num_speakers            (int): number of speakes in array
        speaker_offset          (int): number of channels to offset
        active_speaker          (int): speaker that has signal
        sample                  (float): the actual data

    Returns:
        array of values
    '''
    sample_data = [ 0 for i in range( num_speakers + speaker_offset ) ]
    mask = [0 for i in range(num_speakers + speaker_offset)]
    sample_data[active_speaker + speaker_offset] = sample
    mask[active_speaker + speaker_offset ] = 1.0
    return sample_data , mask

def choose_next_speaker(num_speakers, active_speaker, next_speaker, random_cycling):
    '''
    Picks next active speaker

    Arguments:
        num_speakers            (int): number of speakes in array
        active_speaker          (int): speaker that has signal
        next_speaker            (int): speaker that will next have signal
        random_cycling          (boolean) next speaker is a random one

    Returns:
        array of values
    '''
    next_speaker = random.randint(0, num_speakers -1 ) if random_cycling else ( active_speaker + 1) % (num_speakers)
    while next_speaker == active_speaker:
        next_speaker = random.randint(0, num_speakers - 1)
    return next_speaker


def create_n_channel_data(samples_to_use_shifting, samples_between_chops, speaker_offset, signal, num_speakers, random_cycling = False):
    '''
    Creates data for N channel cycling speaker array

    Arguments:
        samples_to_use_shifting (int): number of samples a shift takes
        samples_to_chop_signal  (int): number of samples between shifts
        speaker_offset          (int): number of channels to offset
        signal                  (numpy.ndarray): the sound signal
        num_speakers            (int): number of speakes in array
                                    (used to avoid monitor outputs)
        random_cycling          (boolean) next speaker is a random one
    Returns: coef1, coef2
        numpy.ndarray corresponding to old speaker channels
    '''
    speaker_data = []
    mask = []
    active_speaker = 0
    next_speaker = -1
    shifting = False
    num_samples_shifting = 0.0
    num_cycles_since_chop = 0

    for sample in signal:
        if( num_cycles_since_chop >= samples_between_chops and not shifting ):
            #picking new active speaker
            next_speaker = choose_next_speaker(num_speakers, active_speaker, next_speaker, random_cycling)
            shifting = True
            num_cycles_since_chop = 0
        if( shifting ):
            if( num_samples_shifting < samples_to_use_shifting ):
                # used during a shift
                speak , mask_sam = create_shifted_sample(num_speakers, speaker_offset, num_samples_shifting,
                samples_to_use_shifting, active_speaker, next_speaker, sample )
                speaker_data.append(speak)
                mask.append(mask_sam)
                num_samples_shifting += 1.0
            else:
                # this means samples shifting has reached its max
                num_samples_shifting = 0
                shifting = False
                active_speaker = next_speaker
                speak , mask_sam = create_regular_sample(num_speakers, speaker_offset, active_speaker, sample)
                speaker_data.append(speak)
                mask.append(mask_sam)
            #need to increment even when shifting
            num_cycles_since_chop += 1
        else:
            # just one speaker is playing
            speak , mask_sam = create_regular_sample(num_speakers, speaker_offset, active_speaker, sample)
            mask.append(mask_sam)
            speaker_data.append(speak)
            num_cycles_since_chop += 1
    return speaker_data , mask


def correctntes_test( original_signal, speaker_data, fs):
    '''
    Takes in original signal and n-channel speaker data to check shifting and
    cycling did not damage the signal.

    Arguments:
        original_signal (numpy.ndarray): original signal data
        speaker_data    (numpy.ndarray): n-channel speaker data
        fs              (numpy.float64): sampling frequency of signal
    '''
    assert len(original_signal) == len(speaker_data)
    print (len(original_signal),len(speaker_data))
    signal_sum = [sum(signals) for signals in speaker_data]
    sf.write('correctntes_test.wav', signal_sum, int(fs))

    for i in range( len( original_signal ) ):
        assert abs( (signal_sum[i] - original_signal[i]) < .00000000001)
    print ('same data')


def random_STAT(a, chunk_size, num_chops):
    '''
    Takes in a signal ans returns it split into num_speakers data whereteh data
    was transformed with an arbitrary transform.

    Arguents:
        a            (np.ndarray): signal (n,)
        chunk_size   (int): size of the transform as well as size that latent chunks will be
        num_chops (int): number of speakers to split signal into
        return_in_playable_format (boolean): True if data should be in sd ready format.
    '''
    orig_len_a = len(a)

    #pad to not loose any signal in splitting
    a = np.hstack( ( a, np.zeros( chunk_size - len(a)%chunk_size ) ) )

    # split for vectorized transform
    a_split = np.array( np.split(a, len(a) // chunk_size ) )

    # create arbitrary transform and modify data with it
    W = np.random.rand(a_split.shape[1],a_split.shape[1])
    transform, R = np.linalg.qr(W)
    a_split_at = np.dot(a_split, transform)


    # prep holder and final
    chops = np.zeros( ( num_chops, a_split_at.shape[0], a_split_at.shape[1] ) ) + 0 * 1j
    final_chops = np.zeros( ( num_chops, len(a)) )

    for i in range(len(a_split_at)):
        # used as a binary mask -- not really needed
        base = np.random.randint(num_chops , size = chops.shape[2])
        for chop in range(num_chops):
            chops[chop][i] = a_split_at[i] * ( (base == chop) + 1j*(base == chop) )

    for chop in range( num_chops ):
        final_chops[chop] =  np.dot(chops[chop],transform.T).flatten().real

    return final_chops[:, :orig_len_a]


def mel_stft(y, chunk_size, num_chops, fs):
    '''
    Takes in a signal and returns it split into num_chops data whereteh data
    was transformed with an fft clipped by a mel bank transform.

    Arguents:
        a            (np.ndarray): signal (n,)
        chunk_size   (int): size of the transform as well as size that latent chunks will be
        num_chops (int): number of speakers to split signal into
        fs (int): Sampling rate of data
    '''
    orig_len_y = len(y)

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_chops + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    fft_bins = np.abs(np.fft.fftfreq( chunk_size ) * fs)
    bin_idx = [(np.abs(fft_bins-hz)).argmin() for hz in hz_points] #+ [chunk_size]


    #pad to not loose any signal in splitting
    y = np.hstack( ( y, np.zeros( chunk_size - len(y)%chunk_size ) ) )

    # split for vectorized transform
    y_split = np.array( np.split(y, len(y) // chunk_size ) )

    # create arbitrary transform and modify data with it
    y_split_at = np.fft.fft(y_split)


    # prep holder and final
    chops = np.zeros( ( num_chops, y_split_at.shape[0], y_split_at.shape[1] ) ) + 0 * 1j
    final_chops = np.zeros( ( num_chops, len(y)) )

    for i in range(len(y_split_at)):
        for chop in range(num_chops):
            chops[chop][i] = np.hstack( ( np.zeros( bin_idx[chop] ),
                                        y_split_at[i][ bin_idx[chop] : bin_idx[chop + 1] ],
                                        np.zeros( chunk_size - bin_idx[chop + 1] ) ) )

    chop_shuffle = np.zeros(chops.shape) + 0 * 1j
    #shuffle to reduce spectral continuity
    for chunk in range(len(chop_shuffle[0])):
        swap = np.arange(len(chop_shuffle))
        np.random.shuffle(swap)
        for chop in range(len(chop_shuffle)):
            chop_shuffle[chop][chunk] = chops[swap[chop]][chunk]

    for chop in range( num_chops ):
        final_chops[chop] =  np.fft.ifft(chop_shuffle[chop]).flatten().real

    return final_chops[:, :orig_len_y]

def gauss_mask(orig_sound, n_speaker, n_signals , length_signal , smooth_length , smooth_std_dev):

    N = length_signal
    mask = np.zeros((n_speaker, n_signals), dtype=object)
    masked_sound = np.empty((n_speaker, n_signals, N))

    # Gaussian masks
    for i in range(n_speaker):
        for j in range(n_signals):

            if i < n_speaker - 1:
                mask[i, j] = np.random.normal((1. / n_speaker), 0.2, (len(orig_sound[j]),))
            else:
                mask[i, j] = np.ones((len(orig_sound[j]),)) - np.sum(mask[:, j])

            # Smoothing out the mask
            mask[i, j] = np.convolve(mask[i, j], sp.signal.gaussian(smooth_length, smooth_std_dev), mode='same')
            masked_sound[i, j] = (orig_sound[j] * mask[i, j]).astype(np.int16)

    speaker_data = np.transpose(masked_sound, axes=[1, 2, 0])

    return speaker_data


if __name__ == "__main__":
    num_samples_shift = 100
    chopping_frequency = 300
    speaker_offset = 2
    num_speakers = 4

    # read in file
    signal, fs = read_wav_file('metalking.wav')
    signal = .07 * signal
    fs = int(fs)

    # creating the data
    speaker_data = create_n_channel_data(num_samples_shift,chopping_frequency, speaker_offset, signal, num_speakers, True)
    #
    # # checking data was preserved
    correctntes_test(signal, speaker_data, fs)

    sd.default.device = 2
    print (sd.query_devices())
    sd.play(speaker_data, fs)
    sd.wait()
