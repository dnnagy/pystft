#include "fftpack.h"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <new> //For std::nothrow
#define PI 3.14159265

/**
  *   TODO: Make a class and release unused memory after calculating STFT and Spectrograms!!!
  */

class FFT {
private:
    int n;
    int* ifac;
    float* wsave;
    float* output;
public:
    FFT(int _n);
    ~FFT();
    float* forward_rfft(float* arr);
};

FFT::FFT(int _n) {
    this->n = _n;
    this->ifac = new int[n+15];
    this->wsave = new float[2*n + 15];
    this->output = new float[n];
};

FFT::~FFT() {
    delete[] this->ifac;
    delete[] this->wsave;
    delete[] this->output;
}

float* FFT::forward_rfft(float* arr) {
    
    // Copy input array to output buffer
    memcpy(output, arr, sizeof(float)*n);
    
    // initialize  rfftf and rfftb
    __ogg_fdrffti(n, wsave, ifac);

    // forward transform of a real periodic sequence
    __ogg_fdrfftf(n, output, wsave, ifac);
    
    return output;
}

// C API for FFT
extern "C" {
    void* create_fft_object(int n){
        return new(std::nothrow) FFT(n);
    }
    
    float* forward_rfft(void* ptr, float* arr) {
        try {
            FFT* ref = reinterpret_cast<FFT *>(ptr);
            return ref->forward_rfft(arr);
        } catch(...) {
            return NULL;
        }
    }
    
    void delete_fft_object(void* ptr) {
        delete ptr;
    }
}


// Helper functions for STFT
extern "C" {
    int calc_nrows(int n, int nperseg, int noverlap){
        return int(floor((float)n/(float)(nperseg-noverlap)))-2;
    }
    int calc_ncols(int n, int nperseg, int noverlap) {
        return int(floor((float)nperseg/2.0f));
    }
    float* get_hann_window(int n) {
        float* buffer = new float[n];

        for(int k=0; k<n; k++){
            buffer[k] = 0.5*(1-cos(2*PI*k/float(n)));
        }        

        return buffer;
    }
}

class STFTPack {
private:
    int n;
    int nperseg;
    int noverlap;
    float eps;
    int padlen = 0;
    
    float* chunk;
    float* chunk_fft;
    float* window;
    float** output;
    
    int nrows;
    int ncols;
    
    FFT* fft_machine;

public:
    STFTPack(int n, int nperseg, int noverlap, float eps);
    ~STFTPack();
    int get_nrows();
    int get_ncols();
    float** forward_stft(float* arr);
    float** forward_spectrogram(float* arr);
    float** forward_log_spectrogram(float* arr);
};

STFTPack::STFTPack(int _n, int _nperseg, int _noverlap, float _eps) {
    
    if (!(_nperseg >= _noverlap && _nperseg%_noverlap == 0)) {
        throw std::invalid_argument(
            "Constraint nperseg >= noverlap && nperseg%noverlap == 0 is not satisfied."
        );
    }
    
    if(_n%_nperseg != 0){
        // We need to pad our array
        this->padlen = _nperseg - _n%_nperseg;
    }
    
    this->n = _n;
    this->nperseg = _nperseg;
    this->noverlap = _noverlap;
    this->eps = _eps;
    
    this->nrows = calc_nrows(_n, _nperseg, _noverlap);
    this->ncols = calc_ncols(_n, _nperseg, _noverlap);
    
    this->window = get_hann_window(_nperseg);
    this->chunk = new float[_nperseg];
    this->chunk_fft = new float[_nperseg];
    
    // This is very inefficient, but straightforward. Could use 1d array instead.
    this->output = new float*[this->nrows];
    for(int j=0; j<this->nrows; j++) {
        this->output[j] = new float[this->ncols];
    }
    
    this->fft_machine = new FFT(nperseg);
}

STFTPack::~STFTPack() {
    for(int j=0; j<this->nrows; j++) {
        delete[] this->output[j];
    }
    delete[] output;
    delete[] window;
    delete[] chunk;
    delete[] chunk_fft;
    delete   fft_machine;
}

int STFTPack::get_nrows() {
    return nrows;
}

int STFTPack::get_ncols() {
    return ncols;
}

float** STFTPack::forward_stft(float* arr) {
    float* x;
    
    if(padlen) {
        x = new float[n + padlen];
        
        // Copy original array
        memcpy(x, arr, sizeof(float)*n);
        
        // Add padding
        for(int k=0; k<padlen; k++){
            x[n+k] = 0.0f;
        }
    } else {
        x = arr;
    }
    
    int k = 0;
    for (int m=0; m<nrows; m++) {
        // chunk*window
        for (int j=0; j<nperseg; j++) {
            chunk[j] = x[k+j]*window[j];
        }
        
        // Result of forward_rfft is destroyed when fft_machine is deallocated
        memcpy(chunk_fft, fft_machine->forward_rfft(chunk), sizeof(float)*nperseg);
            
        for (int j=0; j<ncols; j++){
            output[m][j] = sqrt(
                chunk_fft[2*j]*chunk_fft[2*j] + chunk_fft[2*j+1]*chunk_fft[2*j+1]
            );
        }
        k = k + nperseg - noverlap;
    }
    
    // If array was copied, free the copy
    if (x != arr) {
        delete[] x;
    }
    return this->output;
}

float** STFTPack::forward_spectrogram(float* arr) {
    float** stft = this->forward_stft(arr);
    
    for (int k=0; k<nrows; k++) {
        for (int j=0; j<ncols; j++) {
            stft[k][j] = stft[k][j]*stft[k][j];
        }
    }
        
    return stft;
}

float** STFTPack::forward_log_spectrogram(float* arr) {
    float** stft = this->forward_stft(arr);
        
    for (int k=0; k<nrows; k++) {
        for (int j=0; j<ncols; j++) {
            stft[k][j] = log(eps+stft[k][j]*stft[k][j]);
        }
    }
        
    return stft;
}

// STFTPack C API
extern "C" {
    void* create_stftpack_object(int n, int nperseg, int noverlap, float eps) {
        return new(std::nothrow) STFTPack(n, nperseg, noverlap, eps);
    }
    void delete_stftpack_object(void* ptr) {
        delete ptr;
    }
    float** forward_stft(void* ptr, float* arr) {
        try { 
            STFTPack* ref = reinterpret_cast<STFTPack*>(ptr);
            return ref->forward_stft(arr);
        } catch (...) {
            return NULL;
        }
    }
    float** forward_spectrogram(void* ptr, float* arr) {
        try {
            STFTPack* ref = reinterpret_cast<STFTPack*>(ptr);
            return ref->forward_spectrogram(arr);
        } catch (...) {
            return NULL;
        }
    }
    float** forward_log_spectrogram(void* ptr, float* arr) {
        try {
            STFTPack* ref = reinterpret_cast<STFTPack*>(ptr);
            return ref->forward_log_spectrogram(arr);
        } catch (...) {
            return NULL;
        }
    }
}
