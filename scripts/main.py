import argparse
import numpy as np
import time
from scipy.io.wavfile import read, write
from scripts import denoisify_ss

class CustomHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, max_help_position=200, width=300)


def main():
    parser = argparse.ArgumentParser(
        description="denoisify_ss: Spectral Subtraction-based noise reduction application. "
                    "Work developed by Analía Arimón, Guillermo Mazzeo, and Rodrigo Torrado.",
        usage="spectral-subtraction [options] input_file output_file",
        formatter_class=CustomHelpFormatter
    )

    # Required arguments
    parser.add_argument("input_file", type=str, help="Input file (.wav)")
    parser.add_argument("output_file", type=str, help="Output file (.wav)")

    # General Denoising / Spectral Subtraction parameters
    general_group = parser.add_argument_group("Denoising / Spectral Subtraction")
    general_group.add_argument("--nfft", type=int, default=2048, help="FFT size for STFT analysis and synthesis.")
    general_group.add_argument("--n_iter", type=int, default=30, help="Number of iterations of spectral subtraction.")
    general_group.add_argument("--alpha", type=float, default=0.10, help="Over-subtraction factor.")
    general_group.add_argument("--beta", type=float, default=0.93, help="Spectral floor factor.")
    general_group.add_argument("--rho", type=float, default=0.01, help="Low pass filter smoothing factor.")

    # Noise profile detection parameters
    noise_group = parser.add_argument_group("Noise Profile Detection")
    noise_group.add_argument("--th_energy", type=float, default=0.75, help="Scaling factor for energy-based silence threshold.")
    noise_group.add_argument("--th_zcr", type=float, default=0.35, help="Scaling factor for ZCR-based silence threshold.")
    noise_group.add_argument("--th_he", type=float, default=0.05, help="Scaling factor for high-frequency content threshold.")
    noise_group.add_argument("--zcr_hf_pct_cut", type=float, default=0.90, help="Fraction (0–1) of the ZCR used to define the high-frequency cut-off.")
    noise_group.add_argument("--min_silence_len", type=int, default=10, help="Minimum number of frames to consider a segment as silence.")
    noise_group.add_argument("--min_sound_len", type=int, default=25, help="Minimum number of frames to separate two silence segments.")
    noise_group.add_argument("--start_silence", type=int, default=8, help="Number of frames removed from the start of each detected silence to avoid transients.")
    noise_group.add_argument("--end_silence", type=int, default=1, help="Number of frames removed from the end of each detected silence to avoid transients.")
    noise_group.add_argument("--num_init_frames", type=int, default=5, help="Number of ending frames assumed to be pure noise.")

    # Sinusoidal modeling and transient detection parameters
    spectral_model_group = parser.add_argument_group("Spectral Modeling")
    spectral_model_group.add_argument("--sm_mode", type=int, default=1, help="Spectral modeling mode (0: No modeling, 1: Transients + Sinusoids, 2: Transients only, 3: Sinusoids only).")
    spectral_model_group.add_argument("--sm_keep_pct", type=float, default=0.5, help="Percentage of iterations to retain the spectral model (0 to 1).")
    spectral_model_group.add_argument("--sm_nfft", type=int, default=2048, help="FFT size for sinusoidal modeling.")
    spectral_model_group.add_argument("--peak_thresh", type=int, default=-60, help="Threshold for peak detection in sinusoidal modeling (dB).")
    spectral_model_group.add_argument("--min_sine_dur", type=float, default=0.01, help="Minimum duration of a sinusoid.")
    spectral_model_group.add_argument("--max_sines", type=int, default=100, help="Maximum number of sinusoids.")
    spectral_model_group.add_argument("--fdev_offset", type=int, default=20, help="Offset threshold for frequency deviation in peak continuation.")
    spectral_model_group.add_argument("--fdev_slope", type=float, default=0.01, help="Slope threshold for frequency deviation in peak continuation.")
    spectral_model_group.add_argument("--td_nfft", type=int, default=2048, help="FFT size for transient detection.")
    spectral_model_group.add_argument("--td_Lh", type=float, default=0.9, help="Horizontal median filter length given in seconds or frames.")
    spectral_model_group.add_argument("--td_Lp", type=float, default=1000.0, help="Percussive median filter length given in Hertz or bins.")

    # Musical noise suppression parameters
    musical_noise_group = parser.add_argument_group("Musical Noise")
    musical_noise_group.add_argument("--remove_mn", type=bool, default=False, help="If True, apply musical noise reduction after spectral subtraction.")
    musical_noise_group.add_argument("--mn_nfft", type=int, default=256, help="FFT size for musical noise reduction.")
    musical_noise_group.add_argument("--mn_thresh_db", type=int, default=-25, help="Threshold in dB for spectral floor in musical noise reduction (dB).")
    musical_noise_group.add_argument("--mn_win_len", type=int, default=44, help="Window length for musical noise reduction.")

    # Debugging
    debug_group = parser.add_argument_group("Debugging")
    debug_group.add_argument("--debug", type=int, default=0, help="Print debugging information (0: No, 1: Left Signal, 2: Right Signal, 3: Both).")

    args = parser.parse_args()

    fs, x = read(args.input_file)

    is_stereo = len(x.shape) == 2    
    
    start_time = time.time()

    if is_stereo:
        y_left = denoisify_ss(
            x[:, 0], fs, 
            nfft=args.nfft, n_iter=args.n_iter,
            alpha=args.alpha, beta=args.beta, rho=args.rho, 
            th_energy=args.th_energy, th_zcr=args.th_zcr, th_he=args.th_he,
            zcr_hf_pct_cut=args.zcr_hf_pct_cut, 
            min_silence_len=args.min_silence_len, min_sound_len=args.min_sound_len, 
            start_silence=args.start_silence, end_silence=args.end_silence, 
            num_init_frames=args.num_init_frames,
            sm_mode=args.sm_mode, 
            sm_keep_pct=args.sm_keep_pct,
            sm_nfft=args.sm_nfft, peak_thresh=args.peak_thresh, min_sine_dur=args.min_sine_dur, max_sines=args.max_sines, fdev_offset=args.fdev_offset, fdev_slope=args.fdev_slope,
            td_nfft=args.td_nfft, td_Lh=args.td_Lh, td_Lp=args.td_Lp, 
            remove_mn=args.remove_mn, mn_nfft=args.mn_nfft, mn_thresh_db=args.mn_thresh_db, mn_win_len=args.mn_win_len,
            debug=(args.debug == 1 or args.debug == 3)          
        )

        y_right = denoisify_ss(
            x[:, 1], fs,  
            nfft=args.nfft, n_iter=args.n_iter,
            alpha=args.alpha, beta=args.beta, rho=args.rho, 
            th_energy=args.th_energy, th_zcr=args.th_zcr, th_he=args.th_he,
            zcr_hf_pct_cut=args.zcr_hf_pct_cut, 
            min_silence_len=args.min_silence_len, min_sound_len=args.min_sound_len, 
            start_silence=args.start_silence, end_silence=args.end_silence, 
            num_init_frames=args.num_init_frames,
            sm_mode=args.sm_mode, 
            sm_keep_pct=args.sm_keep_pct,
            sm_nfft=args.sm_nfft, peak_thresh=args.peak_thresh, min_sine_dur=args.min_sine_dur, max_sines=args.max_sines, fdev_offset=args.fdev_offset, fdev_slope=args.fdev_slope,
            td_nfft=args.td_nfft, td_Lh=args.td_Lh, td_Lp=args.td_Lp, 
            remove_mn=args.remove_mn, mn_nfft=args.mn_nfft, mn_thresh_db=args.mn_thresh_db, mn_win_len=args.mn_win_len,
            debug=(args.debug == 2 or args.debug == 3) 
        )

        y = np.column_stack((y_left, y_right))
    else:
        y = denoisify_ss(
            x, fs, 
            nfft=args.nfft, n_iter=args.n_iter,
            alpha=args.alpha, beta=args.beta, rho=args.rho, 
            th_energy=args.th_energy, th_zcr=args.th_zcr, th_he=args.th_he,
            zcr_hf_pct_cut=args.zcr_hf_pct_cut, 
            min_silence_len=args.min_silence_len, min_sound_len=args.min_sound_len, 
            start_silence=args.start_silence, end_silence=args.end_silence, 
            num_init_frames=args.num_init_frames,
            sm_mode=args.sm_mode, 
            sm_keep_pct=args.sm_keep_pct,
            sm_nfft=args.sm_nfft, peak_thresh=args.peak_thresh, min_sine_dur=args.min_sine_dur, max_sines=args.max_sines, fdev_offset=args.fdev_offset, fdev_slope=args.fdev_slope,
            td_nfft=args.td_nfft, td_Lh=args.td_Lh, td_Lp=args.td_Lp, 
            remove_mn=args.remove_mn, mn_nfft=args.mn_nfft, mn_thresh_db=args.mn_thresh_db, mn_win_len=args.mn_win_len,
            debug=(args.debug != 0)
        )

    elapsed_time = time.time() - start_time

    write(args.output_file, fs, np.int16(y))
    
    print(f"Processed file saved to {args.output_file}")
    print(f"Execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
