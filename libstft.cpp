#include "libstft.hpp"

int main(int arc, char* argv[]){
    
    // Generate dummy data
    int N = 16000*60*150;
    
    int nperseg = 128;
    int noverlap = 32;
    float eps = 1e-10;
    
    float* arr = new float[N];
    srand( time(NULL) );
    
    std::cout<< "Generating data of size " << N << " ..."<<std::endl;
    for (int j=0; j<N; j++) {
        arr[j] = -1 + 2*(rand() / (float) RAND_MAX); /* [-1, 1] */
    }
    
    std::cout << "Original array of size " << N << " is at " << &arr << std::endl;
    
    
    // Testing FFT C++ API
    FFT* ptr = new FFT(N);
    std::cout << "Created FFT object at " << ptr << std::endl;
    float* fft = ptr->forward_rfft(arr);
    std::cout << "Calculated FFT is at " << fft << std::endl;
    
    // Creating a copy
    float* result = new float[N];
    memcpy(result, fft, sizeof(float)*N);
    
    std::cout << "Destroying FFT object... " << std::endl;
    delete ptr;
    std::cout << "FFT destroyed. " << std::endl;
    std::cout << "result[N-16] = " << result[N-16] << std::endl;
    delete[] result;
    
    
    // Testing FFT C API
    std::cout << "Testing FFT C API" << std::endl;
    void* cobj = create_fft_object(N);
    std::cout << "FFT objet is at " << cobj << std::endl;
    float* output = forward_rfft(cobj, arr);
    std::cout << "output[N-16] = " << output[N-16] << std::endl;
    delete_fft_object(cobj);
    std::cout << "C API ok. " << std::endl;
    
    // Test STFT C++ API
    STFTPack* stft = new STFTPack(N, nperseg, noverlap, eps);
    std::cout << "Created STFT object at " << stft << std::endl;
    float** to_stft = stft->forward_stft(arr);
    std::cout << "Result of stft is at " << to_stft << std::endl;
    std::cout << "to_stft[7][7] = " << to_stft[7][7] << std::endl;
    std::cout << "Destroying STFT object..." << std::endl;
    delete stft;
    std::cout << "STFT deleted." << std::endl;
    
    stft = new STFTPack(N, nperseg, noverlap, eps);
    std::cout << "Created STFT object at " << stft << std::endl;
    float** specgram = stft->forward_spectrogram(arr);
    std::cout << "Calculated spectrogram is at " << specgram << std::endl;
    std::cout << "specgram[7][7] = " << specgram[7][7] << std::endl;
    float** log_specgram = stft->forward_log_spectrogram(arr);
    std::cout << "Calculated log_spectrogram is at " << log_specgram << std::endl;
    std::cout << "log_specgram[7][7] = " << log_specgram[7][7] << std::endl;
    
    // Test STFT C API
    std::cout << "Testing STFT C API" << std::endl;
    void* stftobj = create_stftpack_object(N, nperseg, noverlap, eps);
    std::cout << "STFT objet is at " << stftobj << std::endl;
    to_stft = forward_stft(stftobj, arr);
    std::cout << "Result of stft is at " << to_stft << std::endl;
    std::cout << "to_stft[7][7] = " << to_stft[7][7] << std::endl;
    specgram = forward_spectrogram(stftobj, arr);
    std::cout << "Calculated spectrogram is at " << specgram << std::endl;
    std::cout << "specgram[7][7] = " << specgram[7][7] << std::endl;
    log_specgram = forward_log_spectrogram(stftobj, arr);
    std::cout << "Calculated log_spectrogram is at " << log_specgram << std::endl;
    std::cout << "log_specgram[7][7] = " << log_specgram[7][7] << std::endl;
    delete_stftpack_object(stftobj);
    std::cout << "STFT C API ok." << std::endl;
    
    std::cout << "Starting heavy log spectrogram testing..." << std::endl;
    
    // Now do a heavy benchmark
    int epochs = 600;
    for (int k=0; k < epochs; k++) {
        std::cout<< "Epoch " << k << " started" << std::endl;
        for (int j=0; j<2500; j++) {
            std::cout<<j<<std::endl;
            STFTPack* ptr = new STFTPack(N, nperseg, noverlap, eps);
            int nrows = ptr->get_nrows();
            int ncols = ptr->get_ncols();
            
            float** output = ptr->forward_log_spectrogram(arr);
            float** result = new float*[nrows];
            for (int j=0; j<nrows; j++) {
                result[j] = new float[ncols];
                memcpy(result[j], output[j], sizeof(float)*ncols);
            }
            
            // Cleanup memory 
            delete ptr;
            for (int j=0; j<nrows; j++) {
                delete[] result[j];
            }
            delete[] result;
        }
    }
    
    return 0;
}
