% ==========================================================
% PROJECT: DIGITAL SIGNAL PROCESSING FUNDAMENTALS
% MATLAB / GNU OCTAVE IMPLEMENTATION
% ==========================================================

clc;
clear;
close all;

% For GNU OCTAVE, uncomment the following line:
% pkg load signal

disp('DIGITAL SIGNAL PROCESSING PROJECT');

%% 1. SIGNAL GENERATION

Fs = 1000;
T = 1/Fs;
t = 0:T:1-T;

f1 = 50;
f2 = 120;

x1 = sin(2*pi*f1*t);
x2 = 0.5*sin(2*pi*f2*t);

x = x1 + x2;

figure(1);

subplot(3,1,1);
plot(t,x1);
title('50 Hz Sine Wave');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(3,1,2);
plot(t,x2);
title('120 Hz Sine Wave');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

subplot(3,1,3);
plot(t,x);
title('Composite Input Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;


%% 2. SAMPLING

figure(2);

stem(t(1:50),x(1:50));
title('Discrete-Time Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;


%% 3. QUANTIZATION

bits = 3;
L = 2^bits;

xmin = min(x);
xmax = max(x);

step = (xmax-xmin)/L;

xq = xmin + step * ...
    floor((x-xmin)/step + 0.5);

xq = min(xq,xmax-step/2);

quant_error = x-xq;

figure(3);

plot(t,x);
hold on;
stairs(t,xq);
title('Signal Quantization');
legend('Original','Quantized');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;


%% 4. TIME SHIFTING

delay = 20;

x_delay = [zeros(1,delay), ...
           x(1:end-delay)];

figure(4);

plot(t,x);
hold on;
plot(t,x_delay);
title('Time Shifting');
legend('Original','Delayed');
grid on;


%% 5. TIME REVERSAL

x_reverse = fliplr(x);

figure(5);

plot(t,x_reverse);
title('Time Reversal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;


%% 6. AMPLITUDE SCALING

x_scaled = 2*x;

figure(6);

plot(t,x_scaled);
title('Amplitude Scaling');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;


%% 7. LINEAR CONVOLUTION

a = [1 2 3 4];
b = [1 1 1];

y_conv = conv(a,b);

disp('Convolution Result:');
disp(y_conv);

figure(7);

stem(y_conv);
title('Linear Convolution');
xlabel('Sample');
ylabel('Amplitude');
grid on;


%% 8. AUTOCORRELATION

y_corr = xcorr(x);

figure(8);

plot(y_corr);
title('Autocorrelation');
xlabel('Lag');
ylabel('Correlation');
grid on;


%% 9. DFT AND FFT

N = length(x);

X = fft(x);

f = (0:N-1)*(Fs/N);

% Single-sided magnitude spectrum

P2 = abs(X/N);

P1 = P2(1:N/2+1);

P1(2:end-1) = 2*P1(2:end-1);

f1_axis = Fs*(0:N/2)/N;

figure(9);

plot(f1_axis,P1);
title('FFT Frequency Spectrum');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

xlim([0 Fs/2]);


%% 10. FIR LOW PASS FILTER

fc = 100;

Wn = fc/(Fs/2);

b_fir = fir1(30,Wn,'low');

y_fir = filter(b_fir,1,x);

figure(10);

subplot(2,1,1);
plot(t,x);
title('Original Signal');
grid on;

subplot(2,1,2);
plot(t,y_fir);
title('FIR Low Pass Filter Output');
grid on;


%% 11. IIR BUTTERWORTH FILTER

[b_iir,a_iir] = butter(4,Wn,'low');

y_iir = filter(b_iir,a_iir,x);

figure(11);

plot(t,y_iir);
title('IIR Butterworth Filter Output');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;


%% 12. FIR FREQUENCY RESPONSE

figure(12);

freqz(b_fir,1,1024,Fs);
title('FIR Frequency Response');


%% 13. IIR FREQUENCY RESPONSE

figure(13);

freqz(b_iir,a_iir,1024,Fs);
title('IIR Frequency Response');


%% 14. NOISE GENERATION

rng(1);

noise = 0.5*randn(size(x));

x_noisy = x+noise;

figure(14);

subplot(2,1,1);
plot(t,x);
title('Original Signal');
grid on;

subplot(2,1,2);
plot(t,x_noisy);
title('Signal with Gaussian Noise');
grid on;


%% 15. NOISE REDUCTION

y_clean = filter(b_fir,1,x_noisy);

figure(15);

plot(t,x_noisy,'Color',[0.7 0.7 0.7]);
hold on;
plot(t,y_clean,'b','LineWidth',1.5);
title('Noise Reduction using FIR Filter');
legend('Noisy Signal','Filtered Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;


%% 16. PERFORMANCE ANALYSIS

input_power = mean(x.^2);

noise_power = mean(noise.^2);

SNR_before = 10*log10(input_power/noise_power);

filtered_error = y_clean-x;

SNR_after = 10*log10( ...
    mean(x.^2)/mean(filtered_error.^2));

quantization_MSE = mean(quant_error.^2);

disp(' ');
disp('======= PERFORMANCE RESULTS =======');

fprintf('Sampling Frequency: %d Hz\n',Fs);

fprintf('Quantization Bits: %d\n',bits);

fprintf('Quantization Levels: %d\n',L);

fprintf('Quantization MSE: %.6f\n', ...
    quantization_MSE);

fprintf('Input SNR: %.2f dB\n',SNR_before);

fprintf('Output SNR: %.2f dB\n',SNR_after);

disp('===================================');

disp('DSP PROJECT EXECUTION COMPLETED');
