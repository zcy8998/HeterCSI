clc; clear; close all;

rng(42);  % 设置随机种子以确保结果可复现
% 数据集配置 (D17-D19)
datasets = {
    % fc(GHz)  K   Δf(kHz) T  Δt(ms) UPA       Scenario          SpeedRange(km/h)
    % {3.5,      32,  180,   16, 0.5,  [4,8],   '3GPP_38.901_UMa_LOS', [30,100]};    % D17
    % {6.7,      64,  180,   24, 1,    [4,4],   '3GPP_38.901_UMi_LOS', [3,50]};      % D18
    % {28,       32,  360,   16, 0.25, [4,8],   '3GPP_38.901_UMa_LOS', [30,100]};    % D19
    {2.1,      32,  120,   16, 0.5,  [4,8],   '3GPP_38.901_UMa_LOS', [120,300]};
    {1.8,      128, 60,    24, 1,    [2,4],   '3GPP_38.901_UMi_NLOS', [3,50]};
};

% 样本配置 - 仅生成测试集
samples_per_dataset = 2000;  % 每个数据集仅生成测试样本

% 创建输出目录
output_dir = '/data/zcy_data/zeroshot';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

for dataset_id = 1:length(datasets)
    % 获取当前数据集配置
    config = datasets{dataset_id};
    fc = config{1} * 1e9;        % 中心频率 (Hz)
    K = config{2};               % 频点数量
    delta_f = config{3} * 1e3;    % 子载波间隔 (Hz)
    T = config{4};                % 时间样本数
    delta_t = config{5} * 1e-3;   % 时间采样间隔 (s)
    UPA = config{6};              % 天线配置 [行, 列]
    scenario = config{7};          % 传播场景
    speed_range = config{8};       % 速度范围 [min, max] (km/h)
    
    disp(['Generating dataset D', num2str(16+dataset_id), ' - ', scenario]);
    
    % 计算天线总数
    Ant = UPA(1) * UPA(2);
    
    % 创建数据集目录
    dataset_dir = fullfile(output_dir, sprintf('D%d', 16+dataset_id)); % D17-D19
    if ~exist(dataset_dir, 'dir')
        mkdir(dataset_dir);
    end
    
    % 预分配测试集存储空间
    H_test = zeros(samples_per_dataset, T, K, Ant, 'single');
    speed_test = zeros(samples_per_dataset, 1);
    
    % 基站位置 (固定)
    BSlocation = [0; 0; 30];  % 基站位置 (x,y,z)
    
    % 用户初始位置范围
    rho_min = 20;   % 最小距离 (m)
    rho_max = 50;   % 最大距离 (m)
    phi_min = -60;  % 最小角度 (度)
    phi_max = 60;   % 最大角度 (度)
    UEcenter = [200; 0; 1.5]; % 用户区域中心
    
    % 仅生成测试样本
    for sample_id = 1:samples_per_dataset
        % 随机生成用户速度
        UESpeed = speed_range(1) + (speed_range(2)-speed_range(1))*rand();
        
        % 创建仿真参数
        s = qd_simulation_parameters;
        s.center_frequency = fc;
        s.set_speed(UESpeed, delta_t);
        s.use_random_initial_phase = true;
        s.use_3GPP_baseline = 1;

        
        % 设置天线阵列
        M_BS = UPA(1);  % 垂直天线数
        N_BS = UPA(2);  % 水平天线数
        ElcTltAgl_BS = 7; % 电下倾角
        
        % 生成基站天线阵列
        BSAntArray = qd_arrayant.generate('3gpp-mmw', M_BS, N_BS,...
            s.center_frequency, 1, ElcTltAgl_BS,...
            0.5, 1, 1, M_BS*0.5, N_BS*0.5);
        
        % 生成用户天线 (单天线)
        UEAntArray = qd_arrayant.generate('3gpp-mmw', 1, 1,...
            s.center_frequency, 1, ElcTltAgl_BS,...
            0.5, 1, 1, 0.5, 0.5);
        
        % 计算用户运动轨迹
        total_time = (T-1) * delta_t;      % 总时间 (s)
        UETrackLength = UESpeed/3.6 * total_time; % 轨迹长度 (m)
        % SnapNum = 1+floor(total_time/delta_t);
        
        % 随机生成用户初始位置
        rho = rho_min + (rho_max-rho_min)*rand();
        phi = phi_min + (phi_max-phi_min)*rand();
        UElocation = [-rho*cosd(phi); rho*sind(phi); 0] + UEcenter;
        
        % 创建用户轨迹
        % step = UETrackLength/(T-1);
        UEtrack = qd_track.generate('linear', UETrackLength);
        UEtrack.name = num2str(sample_id);
        UEtrack.interpolate('distance',1/s.samples_per_meter,[],[],1);
        
        % 创建布局
        l = qd_layout(s);
        l.no_tx = 1;
        l.tx_array = BSAntArray;
        l.tx_position = BSlocation;
        l.no_rx = 1;
        l.rx_array = UEAntArray;
        l.rx_track = UEtrack;
        l.rx_position = UElocation;
        l.set_scenario(scenario);
        
        % 获取信道
        [channel, ~] = l.get_channels();
        
        % 计算CSI (时间×频率×天线)
        bandwidth = K * delta_f;  % 总带宽
        % H_sample = zeros(T, K, Ant, 'single'); % 初始化CSI张量
        

        H_sample = channel.fr(bandwidth, K); % 频域响应
        % disp(['H_sample 维度: ', num2str(size(H_sample))]);
        H_sample = permute(H_sample, [4, 3, 2, 1]); 
        
        % 存储测试样本
        H_test(sample_id, :, :, :) = H_sample;
        speed_test(sample_id) = UESpeed;
        
        % 进度显示
        if mod(sample_id, 100) == 0
            fprintf('Generated %d/%d test samples for D%d\n', sample_id, samples_per_dataset, 19+dataset_id);
        end
    end
    
    % 保存测试集
    test_path = fullfile(dataset_dir, 'test_data.mat');
    save(test_path, 'H_test', 'speed_test', '-v7.3');
    
    % 保存数据集配置
    config_path = fullfile(dataset_dir, 'config.mat');
    save(config_path, 'fc', 'K', 'delta_f', 'T', 'delta_t', 'UPA', 'scenario');
    
    fprintf('Test set for D%d complete. Saved %d samples to %s\n\n', 16+dataset_id, samples_per_dataset, dataset_dir);
    
    clear H_test speed_test  % 清除变量释放内存
end

disp('All test datasets generated successfully!');