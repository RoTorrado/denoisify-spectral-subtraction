import numpy as np
from scripts.noise_detection import noise_profile_detection
from scripts.denoise_signal import denoise_signal
from scripts.utils import print_info_debug

def denoisify_ss(
    x, fs,

    # General spectral subtraction parameters
    nfft,  
    n_iter,
    alpha, beta, rho, 

    # Noise feature detection
    th_energy, th_zcr, th_he,   
    zcr_hf_pct_cut,                                
    min_silence_len, min_sound_len, 
    start_silence, end_silence, 
    num_init_frames,

    # Spectral modeling parameters
    sm_mode,  
    sm_keep_pct,  
    sm_nfft, peak_thresh, min_sine_dur, max_sines, fdev_offset, fdev_slope,
    td_nfft, td_Lh, td_Lp, 

    # Musical noise reduction
    remove_mn, mn_nfft, mn_thresh_db, mn_win_len,
    
    # Debugging
    debug
):
    """
    Apply full denoising pipeline based on spectral subtraction, with
    automatic noise frame detection, optional spectral modeling and 
    musical noise reduction.

    Parameters:
    -----------
    x : ndarray
        Input noisy signal.
    fs : int
        Sampling rate (Hz).
    
    nfft : int
        FFT size for STFT analysis and synthesis.
    n_iter : int
        Number of iterations of spectral subtraction.
    alpha : float
        Over-subtraction factor.
    beta : float
        Spectral floor factor.
    rho : float
        Low pass filter smoothing factor.

    th_energy : float
        Energy threshold for silence detection.
    th_zcr : float
        Zero-crossing rate threshold for silence detection.
    th_he : float
        High-frequency magnitude threshold for silence detection.
    zcr_hf_pct_cut : float
        High-frequency cutoff percentage for ZCR calculation.
    min_silence_len : int
        Minimum duration (frames) for a silence segment.
    min_sound_len : int
        Minimum duration (frames) for a sound segment.
    start_silence : int
        Minimum number of silence frames at the start.
    end_silence : int
        Minimum number of silence frames at the end.
    num_init_frames : int
        Number of initial frames to include as noise reference.

    sm_mode : int
        Spectral modeling mode:
            0 - No modeling,
            1 - Transients + Sinusoids,
            2 - Transients only,
            3 - Sinusoids only.
    sm_keep_pct : float
        Percentage of iterations to retain spectral model (0–1).
    sm_nfft : int
        FFT size for sinusoidal modeling.
    peak_thresh : float
        Threshold for peak detection.
    min_sine_dur : float
        Minimum sinusoid duration (seconds).
    max_sines : int
        Maximum number of sinusoids per frame.
    fdev_offset : float
        Frequency deviation offset for sinusoid continuation.
    fdev_slope : float
        Frequency deviation slope for sinusoid continuation.

    td_nfft : int
        FFT size for transient detection.
    td_Lh : int
        Horizontal median filter length given in seconds or frames.
    td_Lp : int
        Percussive median filter length given in Hertz or bins.

    remove_mn : bool
        Whether to apply musical noise reduction.
    mn_nfft : int
        FFT size for musical noise reduction.
    mn_thresh_db : float
        Threshold (in dB) for musical noise suppression.
    mn_win_len : int
        Smoothing window length for musical noise removal.

    debug : bool
        If True, print debugging information and plots.

    Returns:
    --------
    y : ndarray
        Final denoised output signal.
    """

    x = x.astype(float)

    noise_frames, noise_profile, energy, zcr, hfe_mask = noise_profile_detection(
        x, fs, nfft,
        th_energy, th_zcr, th_he,
        zcr_hf_pct_cut,
        min_silence_len, min_sound_len,
        start_silence, end_silence,
        num_init_frames
    )

    y, x_trans, x_harm, x_sines = denoise_signal(
        x, fs, nfft, 
        alpha, beta, rho,  
        noise_frames,   
        n_iter,
        sm_mode,  
        sm_keep_pct,  
        sm_nfft, peak_thresh, min_sine_dur, max_sines, fdev_offset, fdev_slope, 
        td_nfft, td_Lh, td_Lp, 
        remove_mn, mn_nfft, mn_thresh_db, mn_win_len
    )
    
    if debug:        
        print_info_debug(
            x, y, fs, nfft,
            noise_profile, num_init_frames, noise_frames,
            energy, zcr, hfe_mask,
            th_energy, th_zcr, th_he,
            x_trans, x_harm, x_sines
        )

    return y
