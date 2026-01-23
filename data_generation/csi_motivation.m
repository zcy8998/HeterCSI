% =========================================================================
% Script Name: CSI Dataset Generator (using QuaDRiGa)
% Description: 
%   Generates wireless Channel State Information (CSI) datasets based on 
%   various 3GPP scenarios using the QuaDRiGa channel model.
%   Data is split into Training, Validation, and Test sets.
% =========================================================================

clc; clear; close all;

rng(42);  % Set random seed to ensure reproducibility

datasets = {
    % fc(GHz)  K    Δf(kHz)  T   Δt(ms)  UPA      Scenario                 SpeedRange(km/h)
    {1.5,      128, 90,      24, 1,      [1,4],   '3GPP_38.901_UMi_NLOS',   [3,50]};
    {1.5,      64,  90,      16, 1,      [4,8],   '3GPP_38.901_UMi_NLOS',   [3,50]};
    {4.9,      128, 60,      24, 1,      [1,4],   '3GPP_38.901_Indoor_LOS', [0,10]};
    {2.5,      64,  180,     24, 0.5,    [2,2],   '3GPP_38.901_RMa_NLOS',   [120,300]};
};

% -------------------------------------------------------------------------
% Sampling Configuration
% -------------------------------------------------------------------------
samples_per_dataset = 12000;  % Total number of samples per dataset
train_ratio = 0.75;           % Training set ratio
val_ratio = 0.0833;           % Validation set ratio (1000/12000)
test_ratio = 0.1667;          % Test set ratio (2000/12000)

% Create output directory
% Note: Change this path to your local directory before running
output_dir = './data/csidata/motivation'; 
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

for dataset_id = 1:length(datasets)
    % Retrieve configuration for the current dataset
    config = datasets{dataset_id};
    fc = config{1} * 1e9;         % Center frequency (Hz)
    K = config{2};                % Number of subcarriers (frequency points)
    delta_f = config{3} * 1e3;    % Subcarrier spacing (Hz)
    T = config{4};                % Number of time samples (snapshots)
    delta_t = config{5} * 1e-3;   % Time sampling interval (s)
    UPA = config{6};              % Antenna configuration [Rows, Cols]
    scenario = config{7};         % Propagation scenario
    speed_range = config{8};      % Speed range [min, max] (km/h)
    
    disp(['Processing Dataset ID: ', num2str(dataset_id)])
    disp(['Scenario: ', scenario])
    
    % Calculate total number of antennas
    Ant = UPA(1) * UPA(2);
    
    % Calculate sample splits
    num_train = round(samples_per_dataset * train_ratio);
    num_val = round(samples_per_dataset * val_ratio);
    num_test = samples_per_dataset - num_train - num_val;
    
    % Create directory for the specific dataset
    dataset_dir = fullfile(output_dir, sprintf('D%d', dataset_id));
    if ~exist(dataset_dir, 'dir')
        mkdir(dataset_dir);
    end
    
    % Pre-allocate memory (using single-precision complex numbers to minimize memory usage)
    H_train = zeros(num_train, T, K, Ant, 'single');
    speed_train = zeros(num_train, 1);
    
    H_val = zeros(num_val, T, K, Ant, 'single');
    speed_val = zeros(num_val, 1);
    
    H_test = zeros(num_test, T, K, Ant, 'single');
    speed_test = zeros(num_test, 1);
    
    fprintf('Generating dataset D%d (%d train, %d val, %d test samples)...\n', ...
            dataset_id, num_train, num_val, num_test);
    
    % Base Station (BS) position (Fixed)
    BSlocation = [0; 0; 30];  % (x, y, z) in meters
    
    % User Equipment (UE) initial position range
    rho_min = 20;   % Min distance (m)
    rho_max = 50;   % Max distance (m)
    phi_min = -60;  % Min angle (degrees)
    phi_max = 60;   % Max angle (degrees)
    UEcenter = [200; 0; 1.5]; % Center of the UE area
    
    % Loop to generate samples
    for sample_id = 1:samples_per_dataset
        % Randomly generate UE speed
        UESpeed = speed_range(1) + (speed_range(2)-speed_range(1))*rand();
        
        % Create simulation parameters
        s = qd_simulation_parameters;
        s.center_frequency = fc;
        s.set_speed(UESpeed, delta_t);
        s.use_random_initial_phase = true;
        s.use_3GPP_baseline = 1;

        % Configure antenna arrays
        M_BS = UPA(1);    % Vertical elements
        N_BS = UPA(2);    % Horizontal elements
        ElcTltAgl_BS = 7; % Electrical downtilt angle
        
        % Generate BS antenna array
        BSAntArray = qd_arrayant.generate('3gpp-mmw', M_BS, N_BS,...
            s.center_frequency, 1, ElcTltAgl_BS,...
            0.5, 1, 1, M_BS*0.5, N_BS*0.5);
        
        % Generate UE antenna (Single antenna)
        UEAntArray = qd_arrayant.generate('3gpp-mmw', 1, 1,...
            s.center_frequency, 1, ElcTltAgl_BS,...
            0.5, 1, 1, 0.5, 0.5);
        
        % Calculate UE movement trajectory
        total_time = (T-1) * delta_t;      % Total duration (s)
        UETrackLength = UESpeed / 3.6 * total_time; % Track length (m)
        
        % Randomly generate UE initial position
        rho = rho_min + (rho_max-rho_min)*rand();
        phi = phi_min + (phi_max-phi_min)*rand();
        UElocation = [-rho*cosd(phi); rho*sind(phi); 0] + UEcenter;
        
        % Create UE track
        UEtrack = qd_track.generate('linear', UETrackLength);
        UEtrack.name = num2str(sample_id);
        UEtrack.interpolate('distance', 1/s.samples_per_meter, [], [], 1);
        
        % Create Layout
        l = qd_layout(s);
        l.no_tx = 1;
        l.tx_array = BSAntArray;
        l.tx_position = BSlocation;
        l.no_rx = 1;
        l.rx_array = UEAntArray;
        l.rx_track = UEtrack;
        l.rx_position = UElocation;
        l.set_scenario(scenario);
        
        % Generate channel coefficients
        [channel, ~] = l.get_channels();
        
        % Calculate CSI (Time x Frequency x Antenna)
        bandwidth = K * delta_f;  % Total bandwidth
        
        H_sample = channel.fr(bandwidth, K); % Frequency domain response
        % Permute dimensions to match: [Time, Frequency, Rx, Tx] -> [Time, Freq, Ant]
        % Note: Assumes SISO or MISO where resulting dim is compacted
        H_sample = permute(H_sample, [4, 3, 2, 1]); 
        
        % Determine sample split (train/val/test) and store in the corresponding set
        if sample_id <= num_train
            H_train(sample_id, :, :, :) = H_sample;
            speed_train(sample_id) = UESpeed;
        elseif sample_id <= num_train + num_val
            idx = sample_id - num_train;
            H_val(idx, :, :, :) = H_sample;
            speed_val(idx) = UESpeed;
        else
            idx = sample_id - num_train - num_val;
            H_test(idx, :, :, :) = H_sample;
            speed_test(idx) = UESpeed;
        end
        
        % Display progress
        if mod(sample_id, 100) == 0 % Changed to 100 to reduce console spam
            fprintf('  Generated %d/%d samples for D%d\n', sample_id, samples_per_dataset, dataset_id);
        end
    end
    
    % Save the complete training set
    train_path = fullfile(dataset_dir, 'train_data.mat');
    save(train_path, 'H_train', 'speed_train', '-v7.3');
    
    % Save the complete validation set
    val_path = fullfile(dataset_dir, 'val_data.mat');
    save(val_path, 'H_val', 'speed_val', '-v7.3');
    
    % Save the complete test set
    test_path = fullfile(dataset_dir, 'test_data.mat');
    save(test_path, 'H_test', 'speed_test', '-v7.3');
    
    % Save dataset configuration
    config_path = fullfile(dataset_dir, 'config.mat');
    save(config_path, 'fc', 'K', 'delta_f', 'T', 'delta_t', 'UPA', 'scenario');
    
    fprintf('Dataset D%d generation complete. Saved to %s\n', dataset_id, dataset_dir);
    
    % Clear large variables to free up memory
    clear H_train speed_train H_val speed_val H_test speed_test
end

disp('All datasets generated successfully!');