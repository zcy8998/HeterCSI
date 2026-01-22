% =========================================================================
% Script Name: Zero-Shot CSI Dataset Generator (QuaDRiGa Based)
% Description: 
%   Generates synthetic Channel State Information (CSI) datasets across 
%   diverse 3GPP scenarios (Indoor, RMa, UMa, UMi) to support zero-shot 
%   or generalization learning research.
%
%   Dependencies: Requires QuaDRiGa Channel Model Toolbox.
% =========================================================================

clc; clear; close all;

rng(42);  % Set random seed to ensure reproducibility

% -------------------------------------------------------------------------
% Dataset Configuration
% -------------------------------------------------------------------------
% Configuration Legend (Columns):
% 1. fc (GHz)      : Center Frequency
% 2. K             : Number of Subcarriers
% 3. delta_f (kHz) : Subcarrier Spacing (SCS)
% 4. T             : Number of Time Slots (Snapshots)
% 5. delta_t (ms)  : Time Interval
% 6. UPA           : Antenna Array Config [Rows, Cols]
% 7. Scenario      : 3GPP Propagation Scenario
% 8. SpeedRange    : UE Speed Range [min, max] in km/h

datasets = {
    % --- Indoor Scenarios ---
    {2.4,      64,  30,    16, 1.0,   [4,4],   '3GPP_38.901_Indoor_NLOS', [0,5]};     
    {39,       64,  480,   16, 0.0625,[4,4],   '3GPP_38.901_Indoor_NLOS', [0,10]};   
    {42,       32,  240,   24, 0.125, [2,4],   '3GPP_38.901_Indoor_NLOS', [0,10]};   
    
    % --- RMa Scenarios (Rural Macro) ---
    {2.6,      128, 60,    24, 0.5,   [2,4],   '3GPP_38.901_RMa_LOS',     [100,200]}; 
    {4.8,      128, 240,   24, 0.25,  [2,4],   '3GPP_38.901_RMa_NLOS',    [3,60]};  
    {37.5,     128, 120,   24, 0.25,  [2,4],   '3GPP_38.901_RMa_NLOS',    [3,50]}; 
    
    % --- UMa Scenarios (Urban Macro) ---
    {2.1,      32,  120,   16, 0.5,   [4,8],   '3GPP_38.901_UMa_LOS',     [30,120]};  
    {26.5,     128, 60,    24, 0.5,   [2,8],   '3GPP_38.901_UMa_LOS',     [10,40]};  
    {42,       32,  240,   16, 0.125, [4,8],   '3GPP_38.901_UMa_LOS',     [30,100]}; 
    
    % --- UMi Scenarios (Urban Micro) ---
    {2.1,      128, 360,   24, 1,     [2,4],   '3GPP_38.901_UMi_NLOS',    [3,50]};    
    {6.2,      128, 240,   24, 0.5,   [2,4],   '3GPP_38.901_UMi_NLOS',    [20,80]};   
    {29.0,     128, 120,   24, 0.125, [2,8],   '3GPP_38.901_UMi_LOS',     [30,80]};
};

% -------------------------------------------------------------------------
% Sample Split Configuration
% -------------------------------------------------------------------------
samples_per_dataset = 12000;  % Total samples per dataset
train_ratio = 0.75;           % Training set ratio
val_ratio = 0.0833;           % Validation set ratio (approx. 1000/12000)
test_ratio = 0.1667;          % Test set ratio (approx. 2000/12000)

% Create output directory
output_dir = './data/csidata/zeroshot';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

for dataset_id = 1:length(datasets)
    % Retrieve configuration for the current dataset
    config = datasets{dataset_id};
    fc = config{1} * 1e9;         % Center Frequency (Hz)
    K = config{2};                % Number of Subcarriers
    delta_f = config{3} * 1e3;    % Subcarrier Spacing (Hz)
    T = config{4};                % Number of Time Slots (Snapshots)
    delta_t = config{5} * 1e-3;   % Sampling Interval (s)
    UPA = config{6};              % Antenna Configuration [Rows, Cols]
    scenario = config{7};         % Propagation Scenario
    speed_range = config{8};      % UE Speed Range [min, max] (km/h)
    
    disp(['Processing Dataset ID: ', num2str(dataset_id)])
    disp(['Scenario: ', scenario])
    
    % Calculate total number of antennas
    Ant = UPA(1) * UPA(2);
    
    % Calculate sample splits
    num_train = round(samples_per_dataset * train_ratio);
    num_val = round(samples_per_dataset * val_ratio);
    num_test = samples_per_dataset - num_train - num_val;
    
    % Create subdirectory for the specific dataset
    dataset_dir = fullfile(output_dir, sprintf('D%d', dataset_id));
    if ~exist(dataset_dir, 'dir')
        mkdir(dataset_dir);
    end
    
    % Pre-allocate memory (using single-precision to reduce RAM usage)
    H_train = zeros(num_train, T, K, Ant, 'single');
    speed_train = zeros(num_train, 1);
    
    H_val = zeros(num_val, T, K, Ant, 'single');
    speed_val = zeros(num_val, 1);
    
    H_test = zeros(num_test, T, K, Ant, 'single');
    speed_test = zeros(num_test, 1);
    
    fprintf('Generating dataset D%d (%d train, %d val, %d test samples)...\n', ...
            dataset_id, num_train, num_val, num_test);
    
    % Base Station (BS) Location (Fixed)
    BSlocation = [0; 0; 30];  % (x, y, z) in meters
    
    % User Equipment (UE) Initial Position Range
    rho_min = 20;   % Min Distance (m)
    rho_max = 50;   % Max Distance (m)
    phi_min = -60;  % Min Angle (degrees)
    phi_max = 60;   % Max Angle (degrees)
    UEcenter = [200; 0; 1.5]; % Center of UE area
    
    % Loop to generate samples
    for sample_id = 1:samples_per_dataset
        % Randomly generate UE speed
        UESpeed = speed_range(1) + (speed_range(2)-speed_range(1))*rand();
        
        % Create simulation parameters (QuaDRiGa)
        s = qd_simulation_parameters;
        s.center_frequency = fc;
        s.set_speed(UESpeed, delta_t);
        s.use_random_initial_phase = true;
        s.use_3GPP_baseline = 1;

        % Configure Antenna Array
        M_BS = UPA(1);    % Vertical Elements
        N_BS = UPA(2);    % Horizontal Elements
        ElcTltAgl_BS = 7; % Electrical Downtilt Angle
        
        % Generate BS Antenna Array
        BSAntArray = qd_arrayant.generate('3gpp-mmw', M_BS, N_BS,...
            s.center_frequency, 1, ElcTltAgl_BS,...
            0.5, 1, 1, M_BS*0.5, N_BS*0.5);
        
        % Generate UE Antenna (Single Antenna)
        UEAntArray = qd_arrayant.generate('3gpp-mmw', 1, 1,...
            s.center_frequency, 1, ElcTltAgl_BS,...
            0.5, 1, 1, 0.5, 0.5);
        
        % Calculate UE Trajectory
        total_time = (T-1) * delta_t;      % Total Duration (s)
        UETrackLength = UESpeed / 3.6 * total_time; % Track Length (m)
        
        % Randomly generate UE initial position
        rho = rho_min + (rho_max-rho_min)*rand();
        phi = phi_min + (phi_max-phi_min)*rand();
        UElocation = [-rho*cosd(phi); rho*sind(phi); 0] + UEcenter;
        
        % Generate Linear Track
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
        
        % Generate Channels
        [channel, ~] = l.get_channels();
        
        % Calculate CSI (Time x Frequency x Antenna)
        bandwidth = K * delta_f;  % Total Bandwidth
        
        H_sample = channel.fr(bandwidth, K); % Frequency domain response
        
        % Permute dimensions: [Rx, Tx, Freq, Time] -> [Time, Freq, Tx, Rx] -> [Time, Freq, Ant]
        % Note: Assumes single UE antenna, effectively flattening spatial dim
        H_sample = permute(H_sample, [4, 3, 2, 1]); 
        
        % Distribute samples to Train/Val/Test sets
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
        if mod(sample_id, 1) == 0
            fprintf('  Generated %d/%d samples for D%d\n', sample_id, samples_per_dataset, dataset_id);
        end
    end
    
    % Save Training Data
    train_path = fullfile(dataset_dir, 'train_data.mat');
    save(train_path, 'H_train', 'speed_train', '-v7.3');
    
    % Save Validation Data
    val_path = fullfile(dataset_dir, 'val_data.mat');
    save(val_path, 'H_val', 'speed_val', '-v7.3');
    
    % Save Test Data
    test_path = fullfile(dataset_dir, 'test_data.mat');
    save(test_path, 'H_test', 'speed_test', '-v7.3');
    
    % Save Dataset Configuration
    config_path = fullfile(dataset_dir, 'config.mat');
    save(config_path, 'fc', 'K', 'delta_f', 'T', 'delta_t', 'UPA', 'scenario');
    
    fprintf('Dataset D%d generation complete. Saved to %s\n', dataset_id, dataset_dir);
    
    % Clear large variables to free memory
    clear H_train speed_train H_val speed_val H_test speed_test
end

disp('All datasets generated successfully!');