clear all; close all; clc;

digitnames = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'};

Q = 24;
f_min = 300;
f_max = 8000;

C_coeff_final = cell(9,15);

for digit = 1:9
    for speaker = 1:15
        %% Step 1 
        % skip the missing speakers
        if (((digit==6)&&(speaker==12)))
            continue;
        end
        if ((digit==8)&&(speaker==7))
            continue;
        end
        clear s_o s_p frame freq hamming_window mel_filter Energy G_coeff C_coeff 
        
        audioname = sprintf('./digits2016/%s%d.wav', strjoin(digitnames(digit)), speaker);
        [s_o,fs] = audioread(audioname);
        %sound(s_o,fs); % hear the sounds
        
        %% Step 2 
        %apply Hpreemph  { 1 - 0.97 * z^(-1) }  
        s_p = filter([1,-0.97],[1,0],s_o);
        
        %% Step 3
        % Tframe = 25ms and Toverlap = 10ms (ideal for speech signals)
        
        samples = 0.025 * fs ;          % deigmata ana plasio
        samples_overlapped = 0.01 * fs ; % overlaped samples ana plaisio
        
        % diaxwrismos se plaisia
        frame =  buffer(s_p, samples, samples_overlapped)';
        
        % dhmiourgia kai efarmogh hamming para8urou
        hamming_window = hamming(samples);
        hamming_window = repmat(hamming_window', size(frame,1),1);
        frame = frame.*hamming_window;
        
        %% Step 4
        FFT_samples = 2^nextpow2(samples); 
        %find the frequencies in mel scale to make the linspace as they are
        %are normally distributed in this scale
        mel_low_fr = 2595 * log10 ( 1 + (f_min/700)) ; % j=0
        mel_high_fr = 2595 * log10 ( 1 + (f_max/700)) ; %j=24 = Q+1
        %the frequencies in mel scale are the following:
        %we take min,max only as limits not as frequencies
        mel_freqs = linspace(mel_low_fr,mel_high_fr,Q+2); %mporei na 8elei kai Q sketo
        %the real frequencies in Hz are the following:
        %we simply do the inversion from mel frequencies' formula
        freqs_Hz = (10.^(mel_freqs / 2595) - 1 ) * 700 ; 
        %final calculation of our filter system
        %making freqs_Hz integers for our digital signals:
        freqs_Hz_norm = floor((FFT_samples+1)*freqs_Hz/fs);  
        
        mel_filter = zeros(Q,FFT_samples); %we will construct Q filter, each one with a centric frequency
        %we have Q+2 frequencies so Q linear parts, each one goes to one of
        %our filters
        
        for i=2:Q+1
           %construct the first half of each filter
           for j = freqs_Hz_norm(i-1):freqs_Hz_norm(i)
               mel_filter(i-1,j) = (j-freqs_Hz_norm(i-1))/(freqs_Hz_norm(i)-freqs_Hz_norm(i-1));
           end
           %construct the second half of each filter
           for j = freqs_Hz_norm(i):freqs_Hz_norm(i+1)
               mel_filter(i-1,j) = 1 - (j-freqs_Hz_norm(i))/(freqs_Hz_norm(i+1)-freqs_Hz_norm(i)) ;
           end
        end
        %x = 1:size(mel_filter,2);
        %plot(x,mel_filter);
        %%Step 5
        %calculation of the fft of our frames
        fft_frames = fft(frame',FFT_samples)';
        fft_frames = repmat(fft_frames,Q,1);
        %creating the total filter to apply to all of our frames
        B = ones(size(frame,1),1);
        total_filter = kron(mel_filter,B); %replicate each row as many times as our frames are
        %reference : https://www.mathworks.com/help/matlab/ref/kron.html
        fft_frames_filtered = fft_frames .* total_filter;
        
        Energy = sum(abs(fft_frames_filtered).^2,2)/FFT_samples;
        %plot the energy for 2 different frames
        %we choose 10th and 20th frame of our first audio clip
        if (digit == 1 && speaker == 1)
            Energy_30 = Energy(30:size(frame,1):size(Energy,1));
            Power_Spectrum_30 = 2*abs(fft_frames(30,:)).^2/FFT_samples;%power spectrum calculation for later use 
            Energy_40 = Energy(40:size(frame,1):size(Energy,1));
            Power_Spectrum_40 = 2*abs(fft_frames(40,:)).^2/FFT_samples;%power spectrum calculation for later use 
            figure('Name',sprintf('Energy of frame 30'));
            plot(Energy_30);
            print -djpeg 'frame_30_original.jpg'
            figure('Name',sprintf('Energy of frame 40'));
            plot(Energy_40);
            print -djpeg 'frame_40_original.jpg'
        end  
        
        %%Step 6
        G_coeff = log10(Energy);
        %the first filter is applied to all frames at the first
        %size(frame,1) rows , the second in the next size(frame,1) rows
        %etc.
        
        %%Step 7
        Nc = 13;
        C_coeff = zeros(size(frame,1),Q);
        for i=1:size(frame,1)
            C_coeff(i,:) = dct(G_coeff(i:size(frame,1):size(Energy,1)));
        end
        %save the C coefficients into a cell array
        C_coeff_final{digit,speaker} = C_coeff(:,1:Nc); %take the Nc first samples of DCT         
        %C_coeff_final{digit,speaker} = C_coeff(:,1:Nc)'; %take the Nc first samples of DCT     
    end
end

% Vavouliotis 03112083
k1 = 3; 
k2 = 8; 
n1 = mod(28,Nc); 
n2 = mod(03,Nc); 
%creating the histograms
figure('Name','Histogramms','NumberTitle','off');

subplot(2,2,1);
hist(C_coeff_final{k1,1}(:,n1));
title(sprintf('C(%d),three', n1));

subplot(2,2,2);
hist(C_coeff_final{k1,1}(:,n2));
title(sprintf('C(%d),three', n2));

subplot(2,2,3);
hist(C_coeff_final{k2,1}(:,n1));
title(sprintf('C(%d),eight', n1));

subplot(2,2,4);
hist(C_coeff_final{k2,1}(:,n2));
title(sprintf('C(%d),eight', n2));

print -djpeg 'VavouliotisHist.jpg'

% Stavrakakis 03112017
k1 = 7; 
k2 = 1; 
n1 = mod(21,Nc); 
n2 = mod(07,Nc); 
%creating the histograms
figure('Name','Histogramms','NumberTitle','off');

subplot(2,2,1);
hist(C_coeff_final{k1,1}(:,n1));
title(sprintf('C(%d),seven', n1));

subplot(2,2,2);
hist(C_coeff_final{k1,1}(:,n2));
title(sprintf('C(%d),seven', n2));

subplot(2,2,3);
hist(C_coeff_final{k2,1}(:,n1));
title(sprintf('C(%d),one', n1));

subplot(2,2,4);
hist(C_coeff_final{k2,1}(:,n2));
title(sprintf('C(%d),one', n2));

print -djpeg 'StavrakakisHist.jpg'

%%Step 8
%reconstruction of frames 10 and 20 of our first audio clip
%we will take inverse dct to go to G_coeff and then 10^ to get back to
%Energy
%close all;
G_coeff_reconstr_30 = idct(C_coeff_final{1,1}(30,:),Q); %we want idct of Q points
G_coeff_reconstr_40 = idct(C_coeff_final{1,1}(40,:),Q);
Energy_reconstr_30 = 10.^G_coeff_reconstr_30';
Energy_reconstr_40 = 10.^G_coeff_reconstr_40';

Energy_reconstr_30_synchr = zeros(1,FFT_samples/2);
Energy_reconstr_40_synchr = zeros(1,FFT_samples/2);
for i = 1:Q
    Energy_reconstr_30_synchr(freqs_Hz_norm(i)) = Energy_reconstr_30(i);
    Energy_reconstr_40_synchr(freqs_Hz_norm(i)) = Energy_reconstr_40(i);
end
figure('Name',sprintf('Energy of frame 30'));
hold on; grid on;
plot(Power_Spectrum_30(1:(FFT_samples/2)));
plot(Energy_reconstr_30_synchr,'r');
legend('Power Spectrum', 'Reconstructed Energy');
hold off;
print -djpeg 'Step8-frame30.jpg'

figure('Name',sprintf('Energy of frame 40'));
hold on; grid on;
plot(Power_Spectrum_40(1:(FFT_samples/2)));
plot(Energy_reconstr_40_synchr,'r');
legend('Power Spectrum', 'Reconstructed Energy');
hold off;
print -djpeg 'Step8-frame40.jpg'

%%Step 9 
%the set of our marker specifiers for 9 different classes
marker_specifiers = {'+' 'o' '*' '.' 'x' 'diamond' 'square' '^' 'v'};

color = {[116 1 19]./255 [43 206 7]./255 [254 159 9]./255 [8 28 250]./255 [149 135 126]./255 [229 26 35]./255 [124 43 121]./255 [17 73 27]./255 [225 237 44]./255};

% Vavouliotis 03112083
k1 = 3; 
k2 = 8; 
n1 = mod(28,Nc); 
n2 = mod(03,Nc); 
%Average_total_data = cell(9,15);
%Average_per_digit = cell(9,1);
Average_total_data = cellfun(@mean, C_coeff_final,'UniformOutput', false);
Average_per_digit = Find_mean_from_data(Average_total_data,Nc);

%plot our results
figure('Name','C MeanValues','NumberTitle','off');

%plot 133 values:

for digit=1:9
    cellfun(@(x) plot_my_cells(x,n1,n2,marker_specifiers{digit},color{digit}), Average_total_data(digit,:) ,'UniformOutput', false);
    xlabel(sprintf('MeanValue C(n1=%d)',n1)); ylabel(sprintf('MeanValue C(n2=%d)',n2));
    %plot the mean values per digit for n1 and n2
    cellfun(@(x) plot_my_cells(x,n1,n2,marker_specifiers{digit},color{digit},digitnames{digit}),Average_per_digit(digit,:),'UniformOutput', false);
 end

hold off;
print -djpeg 'Step9MeanValVavouliotis.jpg'

% Stavrakakis 03112017
k1 = 7; 
k2 = 1; 
n1 = mod(21,Nc); 
n2 = mod(07,Nc); 
figure('Name','C MeanValues','NumberTitle','off');
%plot 133 values:
for digit=1:9
    cellfun(@(x) plot_my_cells(x,n1,n2,marker_specifiers{digit},color{digit}), Average_total_data(digit,:) ,'UniformOutput', false);
    xlabel(sprintf('MeanValue C(n1=%d)',n1)); ylabel(sprintf('MeanValue C(n2=%d)',n2));
    %plot the mean values per digit for n1 and n2
    cellfun(@(x) plot_my_cells(x,n1,n2,marker_specifiers{digit},color{digit},digitnames{digit}),Average_per_digit(digit,:),'UniformOutput', false);
end
print -djpeg 'Step9MeanValStavrakakis.jpg'


