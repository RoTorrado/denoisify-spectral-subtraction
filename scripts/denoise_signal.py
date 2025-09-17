import numpy as np
from scripts.signal_processing import analysis_STFT, synthesis_STFT, spectral_subtraction
from scripts.spectral_modeling import sinusoidal_modeling, transient_detection
from scripts.musical_noise import remove_musical_noise

def denoise_signal(
    x, fs, nfft,
    alpha, beta, rho,
    noise_frames,
    n_iter,
    sm_mode,
    sm_keep_pct,
    sm_nfft, threshold, min_dur, max_sines, fdev_offset, fdev_slope,
    td_nfft, td_Lh, td_Lp,
    remove_mn, mn_nfft, mn_thresh_db, mn_win_len
):
    """
    Perform signal denoising using iterative spectral subtraction,
    optionally enhanced by transient detection and sinusoidal modeling.

    Parameters:
    -----------
    x : ndarray
        Input noisy signal.
    fs : int
        Sampling rate (Hz).
    nfft : int
        FFT size for STFT analysis and synthesis.
    alpha : float
        Over-subtraction factor for spectral subtraction.
    beta : float
        Spectral floor parameter for spectral subtraction.
    rho : float
        Smoothing factor for spectral subtraction.
    noise_frames : ndarray
        Indices of noise-only STFT frames used for noise estimation.
    n_iter : int
        Number of spectral subtraction iterations.
    sm_mode : int
        Spectral modeling mode:
            0 - No modeling,
            1 - Transients + Sinusoids,
            2 - Transients only,
            3 - Sinusoids only.
    sm_keep_pct : float
        Percentage of iterations to retain the spectral model (0 to 1).
    sm_nfft : int
        FFT size for sinusoidal modeling.
    threshold : float
        Threshold for peak detection in sinusoidal modeling.
    min_dur : float
        Minimum duration of a sinusoid.
    max_sines : int
        Maximum number of sinusoids.
    fdev_offset : float
        Frequency deviation offset for peak continuation.
    fdev_slope : float
        Frequency deviation slope for peak continuation.
    td_nfft : int
        FFT size for transient detection.
    td_Lh : int
        Horizontal median filter length given in seconds or frames.
    td_Lp : int
        Percussive median filter length given in Hertz or bins.
    remove_mn : bool
        If True, apply musical noise reduction after spectral subtraction.
    mn_nfft : int
        FFT size for musical noise reduction.
    mn_thresh_db : float
        Threshold in dB for spectral floor in musical noise reduction.
    mn_win_len : int
        Window length for musical noise reduction.

    Returns:
    --------
    y : ndarray
        Denoised output signal.
    x_trans : ndarray or None
        Estimated transient component (None if not used).
    x_harm : ndarray or None
        Harmonic signal without transients (None if not used).
    x_sines : ndarray or None
        Sinusoidal model of the input (None if not used).
    """
    
    x_trans = x_harm = x_sines = None

    if sm_mode != 0:
        keep_until = int(np.ceil(sm_keep_pct * n_iter))

        do_td = sm_mode in {1, 2}
        do_sm = sm_mode in {1, 3}

        if do_td:
            _, x_trans = transient_detection(x, fs, td_nfft, td_nfft // 4, td_Lh, td_Lp)
            x_harm = x[:len(x_trans)] - x_trans
            residual = x_harm
            spec_model = x_trans

        if do_sm:
            input_sig = residual if do_td else x

            x_sines, residual = sinusoidal_modeling(
                input_sig, fs, sm_nfft,
                threshold, min_dur, max_sines,
                fdev_offset, fdev_slope
            )

            if do_td:
                min_len = min(len(x_sines), len(x_trans))
                x_trans = x_trans[:min_len]
                x_sines = x_sines[:min_len]
                residual = residual[:min_len]
                spec_model = x_trans + x_sines
            else:
                spec_model = x_sines

        iter_sig = residual
    else:
        iter_sig = x

    X = analysis_STFT(iter_sig, nfft, nfft // 4)

    for i in range(n_iter):
        X = spectral_subtraction(X, alpha, beta, rho, noise_frames)

        if sm_mode != 0 and i == (keep_until - 1):
            iter_sig = synthesis_STFT(X, nfft, nfft // 4)
            X = analysis_STFT(iter_sig + spec_model[:len(iter_sig)], nfft, nfft // 4)

    y = synthesis_STFT(X, nfft, nfft // 4)

    if remove_mn:
        y = remove_musical_noise(y, mn_nfft, mn_nfft // 4, mn_thresh_db, mn_win_len)
    
    return y, x_trans, x_harm, x_sines
