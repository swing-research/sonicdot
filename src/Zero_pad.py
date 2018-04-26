import numpy as np

def front_pad(x , num_zeros):

    '''

    :param x: the sequence we want to pad
    :param num_zeros: Number of zeroes we want to pad to the front
    :return: result: a numpy array which has zeros padded at front
    '''

    l = len(x)
    result = np.zeros((num_zeros + l,))
    result[num_zeros:] = x

    return result

def back_pad(x , num_zeros):

    '''

   :param x: the sequence we want to pad
   :param num_zeros: Number of zeroes we want to pad to the back
   :return: result: a numpy array which has zeros padded at back
   '''

    l = len(x)
    result = np.zeros((num_zeros + l,))
    result[:l] = x

    return result

def both_pad(x , num_zeros):
    '''

   :param x: the sequence we want to pad
   :param num_zeros: Number of zeroes we want to pad to both sides of the sequence x
   :return: result: a numpy array which has zeros padded on both sides
   '''

    l = len(x)
    result = np.zeros((num_zeros + l + num_zeros,))
    result[num_zeros:(num_zeros + l)] = x

    return result



