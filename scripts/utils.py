import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from IPython.display import Audio
from scipy.signal import spectrogram
from scripts.signal_processing import analysis_STFT
import matplotlib.ticker as ticker
import librosa
import librosa.display


def wavread(filename):
    fs, x = read(filename)

    # Pasar de estéreo a mono
    if len(x.shape) > 1:
        x = (np.float32(x[:,0]) + np.float32(x[:,1])) / 2
    else:
        x = np.float32(x)

    return fs, x

def ajustar_ruido(signal, ruido, snr_deseado):
    signal = np.float32(signal)
    ruido = np.float32(ruido)

    potencia_signal = np.mean(signal**2)
    potencia_ruido = np.mean(ruido**2)
    
    factor = np.sqrt(potencia_signal / (10**(snr_deseado / 10)) / potencia_ruido)
    
    return ruido * factor

def print_init(song_dir, noise_dir, duracion, snr_db, nfft):
    if noise_dir is not None:
        hop = nfft // 4

        fs, x = wavread(song_dir)
        fsw, w = wavread(noise_dir)

        print("Audio:", song_dir)
        print("Ruido:", noise_dir)
        print("Audio Sampling Rate:", fs)
        print()

        x = x[-int(duracion * fs):]
        
        if fs != fsw:
            print("WARNING: Different sampling rates between the chosen Audio and Noise.")

        # Ajustar longitud
        if len(x) < len(w):
            w = w[:len(x)]
        else:
            x = x[:len(w)]
            
        # Ajustar ruido según SNR
        w = ajustar_ruido(x, w, snr_db)

        # Señal + Ruido
        xw = x + w

        print("Original Signal")
        display(Audio(x, rate=fs))

        print(f'Noise: SNR = {snr_db} dB')
        display(Audio(w, rate=fs))

        print(f'Noisy Signal: SNR = {snr_db} dB')
        display(Audio(xw, rate=fs))

        # Perfil espectral esperado (magnitud media)
        perfil_esperado = np.mean(np.abs(analysis_STFT(w, nfft, hop)), axis=1)

        plt.figure(figsize=(12, 3))
        plt.plot(perfil_esperado)
        plt.title("Original Noise Profile")
        plt.grid(True)
        plt.show()
        
        # ======== ESPECTROGRAMA EN ESCALA MEL ========
        n_mels = 256  # Número de bandas Mel
        Sxx_w_mel = librosa.feature.melspectrogram(
            y=w, sr=fs, n_fft=nfft, hop_length=hop, n_mels=n_mels, power=2.0
        )

        Sxx_w_db = librosa.power_to_db(Sxx_w_mel)

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        img = librosa.display.specshow(
            Sxx_w_db, sr=fs, hop_length=hop, x_axis='time', y_axis='mel', cmap='viridis'
        )

        ax.set_title('Noise Mel-Spectrogram')
        plt.colorbar(img, ax=ax, format='%+2.0f dB', label='Intensity [dB]')
        plt.tight_layout()
        plt.show()

        return xw, fs

def print_info_debug(
    x, y, fs, nfft, 
    W_P, init_frames, ind_frames, 
    ste, stzcr, hfsmm, 
    fth_e, fth_zcr, fth_he,
    x_p, x_h, xsm
):
    """
    Función para imprimir información de señales, detección de ruido y modelado espectral (en escala mel y mismo rango dB).
    """
    hop = nfft // 4
    n_mels = 128  # número de bandas mel

    def mel_spectrogram_db(sig):
        S = librosa.feature.melspectrogram(y=sig, sr=fs, n_fft=nfft, hop_length=hop, n_mels=n_mels, power=2.0)
        return librosa.power_to_db(S)

    # ================= AUDIO PRINCIPAL =================
    if x is not None and y is not None:
        print("Input Signal")
        display(Audio(x[:len(y)], rate=fs))

        print("Output Signal")
        display(Audio(y, rate=fs))

        print("Residual")
        display(Audio(x[:len(y)] - y, rate=fs))

        # Espectrogramas en escala mel
        Sxx_x_db = mel_spectrogram_db(x[:len(y)])
        Sxx_y_db = mel_spectrogram_db(y)
        Sxx_r_db = mel_spectrogram_db(x[:len(y)] - y)

        # Rango global de dB (misma escala)
        all_Sxx = [Sxx_x_db, Sxx_y_db, Sxx_r_db]
        vmax = max(S.max() for S in all_Sxx)
        vmin = min(S.min() for S in all_Sxx)

        fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharey=True)

        for ax, Sxx_db, title in zip(
            axs,
            [Sxx_x_db, Sxx_y_db, Sxx_r_db],
            ["Input", "Output", "Residual"]
        ):
            img = librosa.display.specshow(
                Sxx_db, sr=fs, hop_length=hop, x_axis='time', y_axis='mel',
                cmap='viridis', ax=ax, vmin=vmin, vmax=vmax
            )
            ax.set_title(f'{title} Mel-Spectrogram (same dB scale)')
            plt.colorbar(img, ax=ax, format='%+2.0f dB', label='Intensity [dB]')

        axs[2].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    # ================= MÉTRICAS =================
    if all(v is not None for v in [fs, hop, W_P, init_frames, ind_frames, ste, stzcr, hfsmm, fth_e, fth_zcr, fth_he]):
        umbrales = [np.mean(arr[-init_frames:]) * (1 + factor) for arr, factor in zip([ste, stzcr, hfsmm], [fth_e, -fth_zcr, fth_he])]
        labels = ['Energy', 'Zero Crossing Rate', 'Spectral Magnitude in High Frequencies']
        
        plt.figure(figsize=(12, 15))
        for i, (arr, umbral, label) in enumerate(zip([ste, stzcr, hfsmm], umbrales, labels)):
            plt.subplot(5, 1, i+1)
            plt.plot(np.arange(len(arr)) * hop / fs, arr, label=label)
            plt.axhline(y=umbral, color='r', linestyle='--', label='Threshold')
            plt.grid(True)
            plt.legend()
            plt.title(label)
        
        plt.subplot(5, 1, 4)
        plt.plot(np.arange(len(ste)) * hop / fs, ste, label='Short Time Energy')
        plt.plot(np.arange(len(ind_frames)) * hop / fs, ind_frames * np.max(ste), label=f'Silence ({np.count_nonzero(ind_frames)} frames)')
        plt.grid(True)
        plt.legend()
        plt.title('Short Time Energy and Silent Frames')

        plt.subplot(5, 1, 5)
        plt.plot(W_P)
        plt.grid(True)
        plt.title("Noise Profile")

        plt.tight_layout()
        plt.show()

    # ================= COMPONENTES ESPECTRALES =================
    if any(v is not None for v in [x_p, x_h, xsm]):
        def plot_components(components, titles):
            """Helper para graficar múltiples espectrogramas mel en misma escala dB"""
            S_list = [mel_spectrogram_db(sig) for sig in components]
            vmax = max(S.max() for S in S_list)
            vmin = min(S.min() for S in S_list)

            fig, axs = plt.subplots(len(S_list), 1, figsize=(12, 4 * len(S_list)), sharey=True)
            if len(S_list) == 1:
                axs = [axs]

            for ax, Sxx_db, title in zip(axs, S_list, titles):
                img = librosa.display.specshow(
                    Sxx_db, sr=fs, hop_length=hop, x_axis='time', y_axis='mel',
                    cmap='viridis', ax=ax, vmin=vmin, vmax=vmax
                )
                ax.set_title(f'{title} Mel-Spectrogram (same dB scale)')
                plt.colorbar(img, ax=ax, format='%+2.0f dB', label='Intensity [dB]')
            axs[-1].set_xlabel("Time (s)")
            plt.tight_layout()
            plt.show()

        if x_p is not None and xsm is not None:
            print("Percussive Component")
            display(Audio(x_p, rate=fs))

            print("Harmonic Component")
            display(Audio(x_h, rate=fs))

            print("Sinusoidal Modeling")
            display(Audio(xsm, rate=fs))

            print("Spectral Modeling")
            display(Audio(xsm + x_p[:len(xsm)], rate=fs))

            print("Spectral Modeling Residual")
            display(Audio(x[:len(xsm)] - x_p[:len(xsm)] - xsm, rate=fs))

            plot_components(
                [x_p[:len(xsm)], x_h[:len(xsm)], xsm],
                ["Transients", "Harmonic", "Sinusoidal"]
            )

        elif xsm is not None:
            print("Sinusoidal Modeling")
            display(Audio(xsm, rate=fs))
            
            print("Modeled Residual")
            display(Audio(x[:len(xsm)] - xsm, rate=fs))
        
        else:
            print("Percussive Component")
            display(Audio(x_p, rate=fs))

            print("Harmonic Component")
            display(Audio(x_h, rate=fs))

            plot_components(
                [x_p, x_h],
                ["Transients", "Harmonic"]
            )



# def print_init_aux(
#     song_dir, noise_dir, duracion, snr_db, nfft
# ):
    
#     if noise_dir is not None:
#         hop = nfft // 4
    
#         fs, x = wavread(song_dir)
#         fsw, w = wavread(noise_dir)

#         print("Audio:", song_dir)
#         print("Ruido:", noise_dir)
#         print("Audio Sampling Rate:", fs)
#         print()

#         x = x[-int(duracion * fs):]
        
#         if fs != fsw:
#             print("WARNING: Different sampling rates between the chosen Audio and Noise.")

#         # Ajustar longitud
#         if len(x) < len(w):
#             w = w[:len(x)]
#         else:
#             x = x[:len(w)]
            
#         # Ajustar ruido según SNR
#         w = ajustar_ruido(x, w, snr_db)

#         # Señal + Ruido
#         xw = x + w

#         print("Original Signal")
#         display(Audio(x, rate=fs))

#         print(f'Noise: SNR = {snr_db} dB')
#         display(Audio(w, rate=fs))

#         print(f'Noisy Signal: SNR = {snr_db} dB')
#         display(Audio(xw, rate=fs))

#         perfil_esperado = np.mean(np.abs(analysis_STFT(w, nfft, hop)), axis=1)

#         plt.figure(figsize=(12, 3))
#         plt.plot(perfil_esperado)
#         plt.title("Original Noise Profile")
#         plt.grid(True)
#         plt.show()
        
#         # Espectrograma del Ruido w
#         f_w, t_w, Sxx_w = spectrogram(w, fs=fs, nperseg=nfft, noverlap=hop)

#         Sxx_w_db = np.maximum(10 * np.log10(Sxx_w + 1e-10), -15)

#         vmin = -15
#         vmax = Sxx_w_db.max()

#         custom_ticks = [10, 100, 1000, 10000]
#         custom_tick_labels = [str(tick) for tick in custom_ticks]

#         fig, ax = plt.subplots(1, 1, figsize=(12, 4))

#         img = ax.pcolormesh(t_w, f_w, Sxx_w_db, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
#         ax.set_title('Noise Spectrogram')
#         ax.set_ylabel('Frequency [Hz]')
#         ax.set_xlabel("Time (s)")
#         ax.set_yscale('log')
#         ax.set_ylim([20, fs / 2])
#         ax.yaxis.set_major_locator(ticker.FixedLocator(custom_ticks))
#         ax.yaxis.set_major_formatter(ticker.FixedFormatter(custom_tick_labels))
#         plt.colorbar(img, ax=ax, orientation='vertical', label='Intensity [dB]')

#         plt.tight_layout()
#         plt.show()


#         return xw, fs
#     else:
#         fs, x = wavread(song_dir)

#         print("Audio:", song_dir)
#         print("Audio Sampling Rate:", fs)
#         print()

#         largo = int(duracion * fs)
#         x = x[-largo:]
        
#         print("Original Signal")
#         display(Audio(x, rate=fs))

#         return x, fs    

# def print_info_debug_aux(
#     x, y, fs, nfft, 
#     W_P, init_frames, ind_frames, 
#     ste, stzcr, hfsmm, 
#     fth_e, fth_zcr, fth_he,
#     x_p, x_h, xsm
# ):
#     """
#     Función para imprimir información de señales, detección de ruido y modelado espectral.
#     """
#     hop = nfft // 4

#     if x is not None and y is not None:
#         print("Input Signal")
#         display(Audio(x[:len(y)], rate=fs))

#         print("Output Signal")
#         display(Audio(y, rate=fs))

#         print("Residual")
#         display(Audio(x[:len(y)] - y, rate=fs))

#         # Espectrogramas
#         f_x, t_x, Sxx_x = spectrogram(x[:len(y)], fs=fs, nperseg=nfft, noverlap=hop)
#         f_y, t_y, Sxx_y = spectrogram(y, fs=fs, nperseg=nfft, noverlap=hop)
#         f_r, t_r, Sxx_r = spectrogram(x[:len(y)] - y, fs=fs, nperseg=nfft, noverlap=hop)

#         # Convertir a dB
#         def to_db(S):
#             return np.maximum(10 * np.log10(S + 1e-10), -15)

#         Sxx_x_db = to_db(Sxx_x)
#         Sxx_y_db = to_db(Sxx_y)
#         Sxx_r_db = to_db(Sxx_r)

#         vmin = -15
#         vmax = max(Sxx_x_db.max(), Sxx_y_db.max(), Sxx_r_db.max())

#         custom_ticks = [10, 100, 1000, 10000]
#         custom_tick_labels = [str(tick) for tick in custom_ticks]

#         # Visualización
#         fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharey=True)

#         for ax, Sxx_db, f, t, title in zip(
#             axs,
#             [Sxx_x_db, Sxx_y_db, Sxx_r_db],
#             [f_x, f_y, f_r],
#             [t_x, t_y, t_r],
#             ["Input", "Output", "Residual"]
#         ):
#             img = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
#             ax.set_title(f'Spectrogram of the {title} Signal')
#             ax.set_ylabel('Frequency [Hz]')
#             ax.set_yscale('log')
#             ax.set_ylim([20, fs / 2])
#             ax.yaxis.set_major_locator(ticker.FixedLocator(custom_ticks))
#             ax.yaxis.set_major_formatter(ticker.FixedFormatter(custom_tick_labels))
#             plt.colorbar(img, ax=ax, orientation='vertical', label='Intensity [dB]')

#         axs[2].set_xlabel("Time (s)")
#         plt.tight_layout()
#         plt.show()

#     if all(v is not None for v in [fs, hop, W_P, init_frames, ind_frames, ste, stzcr, hfsmm, fth_e, fth_zcr, fth_he]):
#         umbrales = [np.mean(arr[-init_frames:]) * (1 + factor) for arr, factor in zip([ste, stzcr, hfsmm], [fth_e, -fth_zcr, fth_he])]
#         labels = ['Energy', 'Zero Crossing Rate', 'Spectral Magnitude in High Frequencies']
        
#         plt.figure(figsize=(12, 15))
#         for i, (arr, umbral, label) in enumerate(zip([ste, stzcr, hfsmm], umbrales, labels)):
#             plt.subplot(5, 1, i+1)
#             plt.plot(np.arange(len(arr)) * hop / fs, arr, label=label)
#             plt.axhline(y=umbral, color='r', linestyle='--', label='Threshold')
#             plt.grid(True)
#             plt.legend()
#             plt.title(label)
        
#         plt.subplot(5, 1, 4)
#         plt.plot(np.arange(len(ste)) * hop / fs, ste, label='Short Time Energy')
#         plt.plot(np.arange(len(ind_frames)) * hop / fs, ind_frames * np.max(ste), label=f'Silence ({np.count_nonzero(ind_frames)} frames)')
#         plt.grid(True)
#         plt.legend()
#         plt.title('Short Time Energy and Silent Frames')

#         plt.subplot(5, 1, 5)
#         plt.plot(W_P)
#         plt.grid(True)
#         plt.title("Noise Profile")

#         plt.tight_layout()
#         plt.show()

#     if any(v is not None for v in [x_p, x_h, xsm]):
#         if x_p is not None and xsm is not None:
#             print("Percussive Component")
#             display(Audio(x_p, rate=fs))

#             print("Harmonic Component")
#             display(Audio(x_h, rate=fs))

#             print("Sinusoidal Modeling")
#             display(Audio(xsm, rate=fs))

#             print("Spectral Modeling")
#             display(Audio(xsm + x_p[:len(xsm)], rate=fs))

#             print("Spectral Modeling Residual")
#             display(Audio(x[:len(xsm)] - x_p[:len(xsm)] - xsm, rate=fs))

#             # Espectrogramas
#             f1, t1, Sxx1 = spectrogram(x_p[:len(xsm)], fs=fs, nperseg=nfft, noverlap=hop)
#             f2, t2, Sxx2 = spectrogram(x_h[:len(xsm)], fs=fs, nperseg=nfft, noverlap=hop)
#             f3, t3, Sxx3 = spectrogram(xsm, fs=fs, nperseg=nfft, noverlap=hop)

#             def to_db(S):
#                 return np.maximum(10 * np.log10(S + 1e-10), -15)

#             Sxx1_db = to_db(Sxx1)
#             Sxx2_db = to_db(Sxx2)
#             Sxx3_db = to_db(Sxx3)

#             vmin = -15
#             vmax = max(Sxx1_db.max(), Sxx2_db.max(), Sxx3_db.max())

#             fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharey=True)

#             for ax, Sxx_db, f, t, title in zip(
#                 axs,
#                 [Sxx1_db, Sxx2_db, Sxx3_db],
#                 [f1, f2, f3],
#                 [t1, t2, t3],
#                 ["Transients", "Harmonic", "Sinusoidal"]
#             ):
#                 img = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
#                 ax.set_title(f'{title} Spectrogram')
#                 ax.set_ylabel('Frequency [Hz]')
#                 ax.set_yscale('log')
#                 ax.set_ylim([20, fs / 2])
#                 ax.yaxis.set_major_locator(ticker.FixedLocator(custom_ticks))
#                 ax.yaxis.set_major_formatter(ticker.FixedFormatter(custom_tick_labels))
#                 plt.colorbar(img, ax=ax, orientation='vertical', label='Intensity [dB]')

#             axs[2].set_xlabel("Time (s)")
#             plt.tight_layout()
#             plt.show()

#         elif xsm is not None:
#             print("Sinusoidal Modeling")
#             display(Audio(xsm, rate=fs))
            
#             print("Modeled Residual")
#             display(Audio(x[:len(xsm)] - xsm, rate=fs))
        
#         else:
#             print("Percussive Component")
#             display(Audio(x_p, rate=fs))

#             print("Harmonic Component")
#             display(Audio(x_h, rate=fs))

#             # Espectrogramas solo para Percusiva y Armónica
#             f1, t1, Sxx1 = spectrogram(x_p, fs=fs, nperseg=nfft, noverlap=hop)
#             f2, t2, Sxx2 = spectrogram(x_h, fs=fs, nperseg=nfft, noverlap=hop)

#             def to_db(S):
#                 return np.maximum(10 * np.log10(S + 1e-10), -15)

#             Sxx1_db = to_db(Sxx1)
#             Sxx2_db = to_db(Sxx2)

#             vmin = -15
#             vmax = max(Sxx1_db.max(), Sxx2_db.max())

#             fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharey=True)

#             for ax, Sxx_db, f, t, title in zip(
#                 axs,
#                 [Sxx1_db, Sxx2_db],
#                 [f1, f2],
#                 [t1, t2],
#                 ["Transients", "Harmonic"]
#             ):
#                 img = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
#                 ax.set_title(f'{title} Spectrogram')
#                 ax.set_ylabel('Frequency [Hz]')
#                 ax.set_yscale('log')
#                 ax.set_ylim([20, fs / 2])
#                 ax.yaxis.set_major_locator(ticker.FixedLocator(custom_ticks))
#                 ax.yaxis.set_major_formatter(ticker.FixedFormatter(custom_tick_labels))
#                 plt.colorbar(img, ax=ax, orientation='vertical', label='Intensity [dB]')

#             axs[1].set_xlabel("Time (s)")
#             plt.tight_layout()
#             plt.show()


    


    


