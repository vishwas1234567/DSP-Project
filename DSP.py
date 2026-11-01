
"""
===============================================================
 PROJECT: DIGITAL SIGNAL PROCESSING FUNDAMENTALS
 IMPLEMENTATION: PYTHON
 APPLICATION: SIGNAL GENERATION, ANALYSIS AND FILTERING
===============================================================

Experiments:
1. Signal Generation
2. Sampling
3. Quantization
4. Time Shifting
5. Time Reversal
6. Amplitude Scaling
7. Linear Convolution
8. Autocorrelation
9. DFT and FFT
10. FIR Low Pass Filtering
11. IIR Butterworth Filtering
12. FIR Frequency Response
13. IIR Frequency Response
14. Noise Generation
15. Noise Reduction
16. Performance Analysis

===============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import (
    firwin,
    butter,
    lfilter,
    freqz,
    fftconvolve
)

# =============================================================
# 1. PROJECT CONFIGURATION
# =============================================================

PROJECT_NAME = "DIGITAL SIGNAL PROCESSING FUNDAMENTALS"

FS = 1000
DURATION = 1.0

FREQ_1 = 50
FREQ_2 = 120

BITS = 3
CUTOFF = 100

RANDOM_SEED = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FIGURE_DIR = os.path.join(BASE_DIR, "results", "figures")
DATA_DIR = os.path.join(BASE_DIR, "results", "data")

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (11, 6),
    "axes.grid": True,
    "grid.alpha": 0.35,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11
})


# =============================================================
# 2. UTILITY FUNCTIONS
# =============================================================

def save_figure(number, name):
    """Save and display the current figure."""

    path = os.path.join(
        FIGURE_DIR,
        f"{number:02d}_{name}.png"
    )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[SAVED] {path}")


def print_section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# =============================================================
# 3. SIGNAL GENERATION
# =============================================================

def generate_signals():

    print_section("1. SIGNAL GENERATION")

    t = np.arange(0, DURATION, 1 / FS)

    x1 = np.sin(2 * np.pi * FREQ_1 * t)

    x2 = 0.5 * np.sin(2 * np.pi * FREQ_2 * t)

    x = x1 + x2

    fig, axes = plt.subplots(3, 1, figsize=(11, 9))

    axes[0].plot(t, x1)
    axes[0].set_title("50 Hz Sine Wave")

    axes[1].plot(t, x2)
    axes[1].set_title("120 Hz Sine Wave")

    axes[2].plot(t, x)
    axes[2].set_title("Composite Input Signal")

    for ax in axes:
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")

    save_figure(1, "signal_generation")

    return t, x1, x2, x


# =============================================================
# 4. SAMPLING
# =============================================================

def sampling(t, x):

    print_section("2. SAMPLING")

    plt.figure()

    plt.stem(t[:50], x[:50])

    plt.title("Discrete-Time Signal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    save_figure(2, "sampling")


# =============================================================
# 5. QUANTIZATION
# =============================================================

def quantization(x):

    print_section("3. QUANTIZATION")

    levels = 2 ** BITS

    xmin = np.min(x)
    xmax = np.max(x)

    step = (xmax - xmin) / levels

    xq = xmin + step * np.floor(
        (x - xmin) / step + 0.5
    )

    xq = np.minimum(xq, xmax - step / 2)

    error = x - xq

    t = np.arange(len(x)) / FS

    plt.figure()

    plt.plot(t, x, label="Original Signal")

    plt.step(
        t,
        xq,
        where="mid",
        label="Quantized Signal"
    )

    plt.title(f"{BITS}-Bit Signal Quantization")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.legend()

    save_figure(3, "quantization")

    return xq, error


# =============================================================
# 6. TIME SHIFTING
# =============================================================

def time_shifting(t, x):

    print_section("4. TIME SHIFTING")

    delay = 20

    x_delay = np.concatenate([
        np.zeros(delay),
        x[:-delay]
    ])

    plt.figure()

    plt.plot(t, x, label="Original")

    plt.plot(t, x_delay, label="Delayed")

    plt.title("Time Shifting")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.legend()

    save_figure(4, "time_shifting")

    return x_delay


# =============================================================
# 7. TIME REVERSAL
# =============================================================

def time_reversal(t, x):

    print_section("5. TIME REVERSAL")

    x_reverse = np.flip(x)

    plt.figure()

    plt.plot(t, x_reverse)

    plt.title("Time Reversal")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    save_figure(5, "time_reversal")

    return x_reverse


# =============================================================
# 8. AMPLITUDE SCALING
# =============================================================

def amplitude_scaling(t, x):

    print_section("6. AMPLITUDE SCALING")

    x_scaled = 2 * x

    plt.figure()

    plt.plot(t, x_scaled)

    plt.title("Amplitude Scaling (Gain = 2)")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    save_figure(6, "amplitude_scaling")

    return x_scaled


# =============================================================
# 9. LINEAR CONVOLUTION
# =============================================================

def linear_convolution():

    print_section("7. LINEAR CONVOLUTION")

    a = np.array([1, 2, 3, 4])

    b = np.array([1, 1, 1])

    y = np.convolve(a, b, mode="full")

    print("Input sequence A:", a)
    print("Input sequence B:", b)

    print("Convolution output:", y)

    plt.figure()

    plt.stem(np.arange(len(y)), y)

    plt.title("Linear Convolution")

    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")

    save_figure(7, "linear_convolution")

    return y


# =============================================================
# 10. AUTOCORRELATION
# =============================================================

def autocorrelation(x):

    print_section("8. AUTOCORRELATION")

    y = np.correlate(x, x, mode="full")

    lags = np.arange(-(len(x) - 1), len(x))

    plt.figure()

    plt.plot(lags, y)

    plt.title("Autocorrelation of Composite Signal")

    plt.xlabel("Lag (samples)")
    plt.ylabel("Correlation")

    save_figure(8, "autocorrelation")

    return y


# =============================================================
# 11. DFT AND FFT
# =============================================================

def fft_analysis(x):

    print_section("9. DFT AND FFT")

    N = len(x)

    X = np.fft.fft(x)

    frequencies = np.fft.rfftfreq(N, 1 / FS)

    magnitude = np.abs(X[:N // 2 + 1]) / N

    if N % 2 == 0:
        magnitude[1:-1] *= 2
    else:
        magnitude[1:] *= 2

    plt.figure()

    plt.plot(frequencies, magnitude)

    plt.title("FFT Frequency Spectrum")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.xlim(0, FS / 2)

    save_figure(9, "fft_spectrum")

    dominant = frequencies[
        np.argsort(magnitude)[-5:]
    ]

    print("Dominant frequency components:", dominant)

    return frequencies, magnitude


# =============================================================
# 12. FIR LOW PASS FILTER
# =============================================================

def fir_filter(x):

    print_section("10. FIR LOW PASS FILTER")

    order = 30

    coefficients = firwin(
        order + 1,
        CUTOFF,
        fs=FS,
        pass_zero="lowpass"
    )

    output = lfilter(coefficients, 1, x)

    t = np.arange(len(x)) / FS

    fig, axes = plt.subplots(2, 1)

    axes[0].plot(t, x)
    axes[0].set_title("Original Signal")

    axes[1].plot(t, output)
    axes[1].set_title("FIR Low Pass Filter Output")

    save_figure(10, "fir_filter")

    return coefficients, output


# =============================================================
# 13. IIR BUTTERWORTH FILTER
# =============================================================

def iir_filter(x):

    print_section("11. IIR BUTTERWORTH FILTER")

    b, a = butter(
        4,
        CUTOFF,
        btype="lowpass",
        fs=FS
    )

    output = lfilter(b, a, x)

    t = np.arange(len(x)) / FS

    plt.figure()

    plt.plot(t, output)

    plt.title("IIR Butterworth Filter Output")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    save_figure(11, "iir_filter")

    return b, a, output


# =============================================================
# 14. FIR FREQUENCY RESPONSE
# =============================================================

def fir_frequency_response(b):

    print_section("12. FIR FREQUENCY RESPONSE")

    f, h = freqz(
        b,
        worN=2048,
        fs=FS
    )

    plt.figure()

    plt.plot(f, 20 * np.log10(
        np.maximum(np.abs(h), 1e-10)
    ))

    plt.title("FIR Frequency Response")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")

    plt.ylim(-100, 5)

    save_figure(12, "fir_frequency_response")


# =============================================================
# 15. IIR FREQUENCY RESPONSE
# =============================================================

def iir_frequency_response(b, a):

    print_section("13. IIR FREQUENCY RESPONSE")

    f, h = freqz(
        b,
        a,
        worN=2048,
        fs=FS
    )

    plt.figure()

    plt.plot(f, 20 * np.log10(
        np.maximum(np.abs(h), 1e-10)
    ))

    plt.title("IIR Butterworth Frequency Response")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")

    plt.ylim(-100, 5)

    save_figure(13, "iir_frequency_response")


# =============================================================
# 16. NOISE GENERATION
# =============================================================

def generate_noise(x):

    print_section("14. NOISE GENERATION")

    rng = np.random.default_rng(RANDOM_SEED)

    noise = 0.5 * rng.standard_normal(len(x))

    noisy = x + noise

    t = np.arange(len(x)) / FS

    fig, axes = plt.subplots(2, 1)

    axes[0].plot(t, x)
    axes[0].set_title("Original Signal")

    axes[1].plot(t, noisy)
    axes[1].set_title("Signal with Gaussian Noise")

    save_figure(14, "noise_generation")

    return noise, noisy


# =============================================================
# 17. NOISE REDUCTION
# =============================================================

def noise_reduction(noisy, b_fir):

    print_section("15. NOISE REDUCTION")

    cleaned = lfilter(b_fir, 1, noisy)

    t = np.arange(len(noisy)) / FS

    plt.figure()

    plt.plot(
        t,
        noisy,
        label="Noisy Signal",
        alpha=0.5
    )

    plt.plot(
        t,
        cleaned,
        label="Filtered Signal",
        linewidth=1.5
    )

    plt.title("Noise Reduction Using FIR Filter")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.legend()

    save_figure(15, "noise_reduction")

    return cleaned


# =============================================================
# 18. PERFORMANCE ANALYSIS
# =============================================================

def performance_analysis(
    x,
    noise,
    cleaned,
    quant_error
):

    print_section("16. PERFORMANCE ANALYSIS")

    input_power = np.mean(x ** 2)

    noise_power = np.mean(noise ** 2)

    snr_before = 10 * np.log10(
        input_power / noise_power
    )

    filtered_error = cleaned - x

    error_power = np.mean(filtered_error ** 2)

    snr_after = 10 * np.log10(
        input_power / error_power
    )

    quantization_mse = np.mean(quant_error ** 2)

    results = {
        "Sampling Frequency (Hz)": FS,
        "Signal Duration (s)": DURATION,
        "Quantization Bits": BITS,
        "Quantization Levels": 2 ** BITS,
        "Input Signal Power": input_power,
        "Noise Power": noise_power,
        "Quantization MSE": quantization_mse,
        "Input SNR (dB)": snr_before,
        "Output SNR (dB)": snr_after,
        "Output Error Power": error_power
    }

    print("\nDSP PERFORMANCE RESULTS\n")

    for key, value in results.items():

        print(f"{key:30s}: {value:.6f}")

    df = pd.DataFrame(
        list(results.items()),
        columns=["Parameter", "Value"]
    )

    csv_path = os.path.join(
        DATA_DIR,
        "dsp_performance_results.csv"
    )

    df.to_csv(csv_path, index=False)

    print("\nPerformance results saved:", csv_path)

    return results


# =============================================================
# 19. SAVE SIGNAL DATA
# =============================================================

def save_signal_data(
    t,
    x,
    xq,
    noisy,
    fir_output,
    iir_output,
    cleaned
):

    data = pd.DataFrame({
        "Time": t,
        "Original Signal": x,
        "Quantized Signal": xq,
        "Noisy Signal": noisy,
        "FIR Output": fir_output,
        "IIR Output": iir_output,
        "Noise Reduced Signal": cleaned
    })

    path = os.path.join(
        DATA_DIR,
        "dsp_signal_data.csv"
    )

    data.to_csv(path, index=False)

    print("\nSignal data saved:", path)


# =============================================================
# 20. MAIN EXECUTION
# =============================================================

def main():

    print("\n" + "=" * 65)
    print(PROJECT_NAME)
    print("PYTHON DIGITAL SIGNAL PROCESSING IMPLEMENTATION")
    print("=" * 65)

    # 1. Signal Generation
    t, x1, x2, x = generate_signals()

    # 2. Sampling
    sampling(t, x)

    # 3. Quantization
    xq, quant_error = quantization(x)

    # 4. Time Shifting
    x_delay = time_shifting(t, x)

    # 5. Time Reversal
    x_reverse = time_reversal(t, x)

    # 6. Amplitude Scaling
    x_scaled = amplitude_scaling(t, x)

    # 7. Linear Convolution
    y_conv = linear_convolution()

    # 8. Autocorrelation
    y_corr = autocorrelation(x)

    # 9. FFT
    frequencies, magnitude = fft_analysis(x)

    # 10. FIR Filter
    b_fir, y_fir = fir_filter(x)

    # 11. IIR Filter
    b_iir, a_iir, y_iir = iir_filter(x)

    # 12. FIR Frequency Response
    fir_frequency_response(b_fir)

    # 13. IIR Frequency Response
    iir_frequency_response(b_iir, a_iir)

    # 14. Noise Generation
    noise, noisy = generate_noise(x)

    # 15. Noise Reduction
    cleaned = noise_reduction(noisy, b_fir)

    # 16. Performance Analysis
    results = performance_analysis(
        x,
        noise,
        cleaned,
        quant_error
    )

    # Save complete signal data
    save_signal_data(
        t,
        x,
        xq,
        noisy,
        y_fir,
        y_iir,
        cleaned
    )

    print_section("PROJECT EXECUTION COMPLETED")

    print("Total experiments completed: 16")

    print("Figures directory:", FIGURE_DIR)

    print("Data directory:", DATA_DIR)

    print("\nAll DSP results generated successfully!")

    plt.show()


if __name__ == "__main__":
    main()
