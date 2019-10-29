%% Computer Assignment (2) ---- Wireless Communication 
%%## Part (1)
close all; clear; clc;

%% Paramaeter Initialization

SNR_dB = 1:10; SNR = 10.^(SNR_dB/10); M = 2; Num_of_OFDM_Block = 64;
Error_Probablity = zeros(1,length(SNR_dB));
Itr_max = 1e5;

%% Simulation

for i = 1 : length(SNR_dB)
    
    Error_Prob_temp = 0;
    for j = 1 : Itr_max  
        data = randi([0 M-1], Num_of_OFDM_Block, 1);
        tx_sig = sqrt(SNR(i)) * pskmod(data,M);
        A = ifft(tx_sig, Num_of_OFDM_Block);

        noise = 1 /sqrt(2) * randn(size(data)) + 1i * 1 / sqrt(2) * randn(size(data));

        y = A + noise;
        Y = fft(y, Num_of_OFDM_Block);

        rx = pskdemod(Y, M);
        num_errors = symerr(data, rx);
        
%         error = data - demod_Sig ;
%         num_of_errors = sum(abs(error));
        
        Error_Prob_temp = Error_Prob_temp + num_errors / Num_of_OFDM_Block;
    end
    Error_Probablity(i) = Error_Prob_temp / Itr_max;
end


%% Result

semilogy(SNR_dB , Error_Probablity);
xlabel('SNR (dB)');
ylabel('Bit Error Rate');
grid on ;
title('OFDM BER Performance');

%%



    
