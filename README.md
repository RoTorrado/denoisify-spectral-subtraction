# SS Denoisify: A structured implementation of spectral subtraction for audio restoration.

This project implements an **audio signal restoration** technique based on *spectral subtraction*, ideal for removing background noise from musicological recordings.

**Denoisify** introduces an **iterative variant of spectral subtraction**, incorporating advanced techniques such as **spectral modeling** through **sinusoidal modeling** and **transient detection**. These methods improve the accuracy of noise removal while preserving the integrity of the original signal. Additionally, the implementation includes an algorithm for **musical noise reduction**, addressing common artifacts typically introduced by basic spectral subtraction.

---

## 🚀 Features

- Iterative spectral subtraction.
- Transient and sinusoidal signal modeling.
- Musical noise suppression.
- Support for mono and stereo WAV files.
- Python-based implementation.
- Interactive usage via Jupyter Notebook.
- Visualization of waveforms and spectra.

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RoTorrado/denoisify-spectral-subtraction.git
   cd denoisify-spectral-subtraction
   
2. **Create and activate a conda environment**:
    ```bash
    conda create -n denoisify python
    conda activate denoisify

3. **Install the dependencies**:
    ```bash
   pip install .
   
4. **Associate the conda environment with Jupyter Notebook:**

    To use the **`denoisify`** environment as a kernel in Jupyter Notebook, run the following commands:

    ```bash
    conda install ipykernel
    python -m ipykernel install --user --name=denoisify --display-name "Python (denoisify)"    
   
## 📓 Example of Use

To demonstrate how to use **Denoisify - Spectral Subtraction**, an example Jupyter notebook, **`Example_Denoisify_SS.ipynb`**, is provided in this repository. The notebook guides you through the process of applying the **spectral subtraction** technique to a noisy audio signal. It includes the following steps:

1. **Loading Audio Files**: It loads both the noisy audio signal and the noise source.
2. **Parameter Adjustments**: You can modify parameters like FFT size, overlap, and thresholds to fine-tune the denoising process.
3. **Visualizing Signals**: The notebook visualizes both the waveform and spectrograms of the original, noisy, and denoised signals.
4. **Signal Processing**: It demonstrates how to separate transients, model sinusoidal components, and apply spectral subtraction.
5. **Listening to Results**: It allows you to listen to intermediate components (like the transient, sinusoidal, and modeled noise) as well as the final denoised output.

The **Jupyter notebook** provides an easy-to-follow workflow for using this technique in various audio signal restoration tasks.

If you prefer to run **Denoisify - Spectral Subtraction** directly from the command line, you can execute:

    spectral-subtraction --help

This will display the available options and arguments. The basic syntax for running the tool is:
    
    spectral-subtraction [options] input_file output_file

Both **input_file** and **output_file** must be **.wav** audio files. The tool supports both mono and stereo formats.

---

### 🎧 Audio and Noise Source

The audio signal used in the demonstration is **"Mother"** by Pink Floyd, sampled at **44.1 kHz**.

The **noise signal** was recorded from a **Revox A77** analog tape recorder, also sampled at **44.1 kHz**. This noise source was obtained from the following academic research:

> I. Irigaray, M. Rocamora, and L. W. P. Biscainho,  
> *Noise reduction in analog tape audio recordings with deep learning models*,  
> AES International Conference on Audio Archiving, Preservation & Restoration,  
> Culpeper, VA, USA, June 2023.

This noise signal serves as a challenging background noise that demonstrates the efficacy of the **Denoisify** spectral subtraction algorithm.

---

### 📄 License

This project is provided for **academic and research purposes only**. The software is distributed under the terms specified in the `LICENSE` file in the repository. Please make sure to review the license before using it in any production environment or redistribution.

---

### 👥 Authors

This project was developed by:

- **Analía Arimón**  
- **Guillermo Mazzeo**  
- **Rodrigo Torrado**

Their contributions have focused on the design, implementation, and optimization of the **Denoisify** algorithm, as well as the preparation of academic documentation and resources.

