import matplotlib as mpl
from matplotlib import pyplot as plt

import numpy as np
from numpy.fft import rfft

import scipy
from scipy.signal import spectrogram

import ctypes
from ctypes import c_float, c_void_p
from ctypes import cdll
from ctypes import POINTER

libstft2 = cdll.LoadLibrary('./libstftv2.so')

# FFT
libstft2.create_fft_object.restype = c_void_p
libstft2.forward_rfft.restype = POINTER( c_float )

# STFT
libstft2.create_stftpack_object.restype = c_void_p
libstft2.forward_stft.restype = POINTER( POINTER( c_float ) )
libstft2.forward_spectrogram.restype = POINTER( POINTER( c_float ) )
libstft2.forward_log_spectrogram.restype = POINTER( POINTER( c_float ) )

# Sample data
nsec = 60
bitrate = 16000

N = int(nsec*bitrate)
nperseg = 1024
noverlap = 512
eps = 1e-10

x = np.linspace(0, 8*np.pi, N)
y = np.sum([
    np.cos(1*k*np.pi*x)+ np.cos(2*k*np.pi*x) +\
    np.cos(3*k*np.pi*x) + np.cos(4*k*np.pi*x) +\
    np.cos(5*k*np.pi*x) + np.cos(6*k*np.pi*x)
    for k in range(1, 28)
], axis=0)

# Normalize
y = y/np.max(y)

# Convert to C array
c_arr = (c_float*len(y))(*y)

# Calculate matrix sizes
nrows = libstft2.calc_nrows(int(len(y)), nperseg, noverlap)
ncols = libstft2.calc_ncols(int(len(y)), nperseg, noverlap)

print("nrows =", nrows, "ncols =", ncols)

# C++ spectrogram
stftobj = libstft2.create_stftpack_object(N, nperseg, noverlap, c_float(eps))
print("STFT object created at", stftobj)
ptr = libstft2.forward_spectrogram(c_void_p(stftobj), c_arr)
print("Result matrix is at", ptr)
np_result = np.stack([ptr[k][:ncols] for k in range(nrows)], axis=1)
np_result = np_result.T
print("Numpy result created")
libstft2.delete_stftpack_object(c_void_p(stftobj))
print("STFT object destroyed")
print("np_result[nrows-10][ncols-10] =", np_result[nrows-10][ncols-10])
print("Done.")