import numpy as np
import os

import ctypes
from ctypes import c_float, c_void_p
from ctypes import POINTER
from ctypes import cdll

mypath = os.path.dirname(__file__)
libpath = os.path.join(mypath, "libstft.so")
libstft = cdll.LoadLibrary( libpath )

# FFT
libstft.create_fft_object.restype = c_void_p
libstft.forward_rfft.restype = POINTER( c_float )

# STFT & Spectrograms
libstft.create_stftpack_object.restype = c_void_p
libstft.forward_stft.restype = POINTER( POINTER( c_float ) )
libstft.forward_spectrogram.restype = POINTER( POINTER( c_float ) )
libstft.forward_log_spectrogram.restype = POINTER( POINTER( c_float ) )

def get_spectrogram(y, nperseg, noverlap, eps):
    """ Returns a 2d numpy array containing the spectrogram of the input. """
    c_arr = (c_float*len(y))(*y)
    
    # Calculate matrix sizes
    nrows = libstft.calc_nrows(int(len(y)), nperseg, noverlap)
    ncols = libstft.calc_ncols(int(len(y)), nperseg, noverlap)
    
    # Do the calculations
    stftobj = libstft.create_stftpack_object(int(len(y)), nperseg, noverlap, c_float(eps))
    ptr = libstft.forward_spectrogram(c_void_p(stftobj), c_arr)
    np_result = np.stack([ptr[k][:ncols] for k in range(nrows)], axis=1)
    
    # Clean up memory
    libstft.delete_stftpack_object(c_void_p(stftobj))
    
    return np_result

def get_log_spectrogram(y, nperseg, noverlap, eps):
    """ Returns a 2d numpy array containing the log spectrogram of the input. """
    c_arr = (c_float*len(y))(*y)
    
    # Calculate matrix sizes
    nrows = libstft.calc_nrows(int(len(y)), nperseg, noverlap)
    ncols = libstft.calc_ncols(int(len(y)), nperseg, noverlap)
    
    # Do the calculations
    stftobj = libstft.create_stftpack_object(int(len(y)), nperseg, noverlap, c_float(eps))
    ptr = libstft.forward_log_spectrogram(c_void_p(stftobj), c_arr)
    np_result = np.stack([ptr[k][:ncols] for k in range(nrows)], axis=1)
    
    # Clean up memory
    libstft.delete_stftpack_object(c_void_p(stftobj))
    
    return np_result
