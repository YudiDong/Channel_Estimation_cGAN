clear all;close all;clc

%% Settings 
bs_ant= 64; % M= 64 BS Antennas
users = 32; % K= 32 Users
pilot_l = 8; % Pilots length is 8
snr  = 0; % SNR = 0 dB

filename = ['Indoor2p4_',num2str(bs_ant),'ant_',num2str(users),'users_',num2str(pilot_l),'pilot'];

%% Generate channel dataset H

%------Ray-tracing scenario
params.scenario='I1_2p4';                % The adopted ray tracing scenarios [check the available scenarios at www.aalkhateeb.net/DeepMIMO.html]

%------parameters set
%Active base stations 
params.active_BS=32;          % Includes the numbers of the active BSs (values from 1-18 for 'O1')

% Active users
params.active_user_first=1;       % The first row of the considered receivers section (check the scenario description for the receiver row map)
params.active_user_last=11;        % The last row of the considered receivers section (check the scenario description for the receiver row map)

% Number of BS Antenna 
params.num_ant_x=1;                  % Number of the UPA antenna array on the x-axis 
params.num_ant_y=bs_ant;                 % Number of the UPA antenna array on the y-axis 
params.num_ant_z=1;                  % Number of the UPA antenna array on the z-axis
                                     % Note: The axes of the antennas match the axes of the ray-tracing scenario
                              
% Antenna spacing
params.ant_spacing=.5;               % ratio of the wavelnegth; for half wavelength enter .5        

% System bandwidth
params.bandwidth=0.01;                % The bandiwdth in GHz 

% OFDM parameters
params.num_OFDM=users;                % Number of OFDM subcarriers
params.OFDM_sampling_factor=1;   % The constructed channels will be calculated only at the sampled subcarriers (to reduce the size of the dataset)
params.OFDM_limit=params.num_OFDM;                % Only the first params.OFDM_limit subcarriers will be considered when constructing the channels

% Number of paths
params.num_paths=10;                  % Maximum number of paths to be considered (a value between 1 and 25), e.g., choose 1 if you are only interested in the strongest path

params.saveDataset=0;
 
% -------------------------- Dataset Generation -----------------%
[DeepMIMO_dataset,params]=DeepMIMO_generator(params); % Get H (i.e.,DeepMIMO_dataset )


%% Generate Pilots 
pilot = uniformPilotsGen(pilot_l);
pilot = pilot{1,1};
pilot_user = repmat(pilot,[1 users])';


%% Genrate Quantized Siganl Y with Noise
channels = zeros(length(DeepMIMO_dataset{1}.user),bs_ant,users);
Y = zeros(length(DeepMIMO_dataset{1}.user),bs_ant,pilot_l);
Y_noise = zeros(length(DeepMIMO_dataset{1}.user),bs_ant,pilot_l);
Y_sign = zeros(length(DeepMIMO_dataset{1}.user),bs_ant,pilot_l,2);

for i = 1:length(DeepMIMO_dataset{1}.user)
channels(i,:,:) = normalize(DeepMIMO_dataset{1}.user{i}.channel,'scale');%%
Y(i,:,:) = DeepMIMO_dataset{1}.user{i}.channel*pilot_user;
Y_noise(i,:,:) = awgn(Y(i,:,:),snr,'measured'); 
end


%% Convert complex data to two-channel data
Y_sign(:,:,:,1) = sign(real(Y_noise)); % real part of Y
Y_sign(:,:,:,2) = sign(imag(Y_noise)); % imag papt of Y


channels_r(:,:,:,1) = real(channels); % real part of H
channels_r(:,:,:,2) = imag(channels); % imag part of H

% Shuffle data 
shuff = randi([1,length(DeepMIMO_dataset{1}.user)],length(DeepMIMO_dataset{1}.user),1);
Y_sign = Y_sign(shuff,:,:,:);
channels_r = channels_r(shuff,:,:,:);


%% Split data for training
numOfSamples = length(DeepMIMO_dataset{1}.user);
trRatio = 0.7;
numTrSamples = floor( trRatio*numOfSamples);
numValSamples = numOfSamples - numTrSamples;

input_da = Y_sign(1:numTrSamples,:,:,:);
output_da = channels_r(1:numTrSamples,:,:,:);

input_da_test = Y_sign(numTrSamples+1:end,:,:,:);
output_da_test = channels_r(numTrSamples+1:end,:,:,:);


%% Visualization of Y and H
figure
imshow(squeeze(input_da(1,:,:,1)))
title('Visualization of Y')
figure
imshow(squeeze(output_da(1,:,:,1)))
title('Visualization of H')

%% Save data
save(['Gan_Data/Gan_',num2str(snr),'_dB',filename],'input_da','output_da','input_da_test','output_da_test','-v7.3');



