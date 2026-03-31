%% CALCE2 数据集 4D 物理特征批量自动化提取与清洗脚本
clear; clc;

% 1. 设置基础路径和要处理的电池编号 (已指向 CALCE2)
basePath = 'D:\OneDrive\桌面\杂\毕设论文\电池充放电数据\CALCE2\'; 
battery_names = {'CS2_35', 'CS2_36', 'CS2_37', 'CS2_38'};

% 初始化图形窗口
figure('Name', 'CALCE 多电池 4D特征(容量)衰减对比', 'Position', [100, 100, 900, 600]);
hold on; grid on;
colors = {'#0072BD', '#D95319', '#77AC30', '#7E2F8E'};

save_vars = {};

%% 2. 开始循环处理每个电池
for b = 1:length(battery_names)
    curr_batt = battery_names{b};
    folderPath = fullfile(basePath, curr_batt);
    
    fprintf('\n🚀 正在处理: %s 并提取 4D IC 特征...\n', curr_batt);
    
    fileList = dir(fullfile(folderPath, '*.xlsx')); 
    if isempty(fileList)
        fprintf('⚠️ 未找到 %s 的数据，跳过。\n', curr_batt);
        continue; 
    end
    
    % 智能日期排序
    fileDates = NaT(length(fileList), 1); 
    pattern = sprintf('%s_(\\d+)_(\\d+)_(\\d+)\\.xlsx', curr_batt);
    for i = 1:length(fileList)
        tokens = regexp(fileList(i).name, pattern, 'tokens');
        if ~isempty(tokens)
            match = tokens{1};
            fileDates(i) = datetime(str2double(match{3})+2000, str2double(match{1}), str2double(match{2}));
        else
            fileDates(i) = datetime(2099, 12, 31); 
        end
    end
    [~, sortIdx] = sort(fileDates);
    fileList = fileList(sortIdx); 

    % 预分配临时特征矩阵 (4维)
    raw_capacity = []; 
    raw_features = []; % [Capacity, End_Voltage, IC_Peak, IC_Vol]
    
    for i = 1:length(fileList)
        full_path = fullfile(fileList(i).folder, fileList(i).name);
        try
            sheets = sheetnames(full_path);
            target_sheet = sheets(contains(sheets, 'Channel', 'IgnoreCase', true));
            if isempty(target_sheet), continue; end
            
            data = readtable(full_path, 'Sheet', target_sheet(1), 'VariableNamingRule', 'preserve');
            
            % 提取放电步 (Step 7) 
            idx = (data.("Step_Index") == 7);
            cycles = unique(data.("Cycle_Index")(idx));
            
            for c = 1:length(cycles)
                c_idx = (data.("Cycle_Index") == cycles(c)) & idx;
                t = data.("Test_Time(s)")(c_idx);
                I = data.("Current(A)")(c_idx);
                % 提取电压序列
                V_seq = data.("Voltage(V)")(c_idx);
                
                if length(t) > 10 % 确保数据点足够提取 IC
                    % 计算总容量和累积容量序列
                    cap = trapz(t, abs(I)) / 3600; 
                    Q_seq = cumtrapz(t, abs(I)) / 3600;
                    
                    if cap > 0.4
                        raw_capacity = [raw_capacity; cap];
                        
                        % 调用你保存的 extract_IC_features 函数
                        try
                            [ic_peak, ic_vol] = extract_IC_features(V_seq, Q_seq);
                        catch
                            % 防御性编程：如果当前圈噪声太大报错，赋安全值
                            ic_peak = 0; ic_vol = 0;
                        end
                        
                        % 记录 4D 特征
                        end_vol = V_seq(end);
                        raw_features = [raw_features; cap, end_vol, ic_peak, ic_vol];
                    end
                end
            end
        catch
            fprintf('读取文件 %s 时出错，跳过。\n', fileList(i).name);
            continue; 
        end
    end
    
    %% 3. 数据归一化、去噪与动态变量生成
    if ~isempty(raw_capacity)
        % 对目标标签 (Y) 进行归一化和平滑
        norm_cap = raw_capacity / raw_capacity(1);
        Y_label = smoothdata(norm_cap, 'movmedian', 15); % 列向量
        cycles_vec = (0:(length(Y_label)-1))';
        
        % 对特征矩阵 (X) 进行归一化 (每一列独立归一化到 0-1) 和轻度平滑
        % 归一化对 GRU 极其重要，防止 IC 峰值和电压尺度差异撕裂梯度
        X_matrix = raw_features;
        for col = 1:4
            min_val = min(X_matrix(:, col));
            max_val = max(X_matrix(:, col));
            if max_val > min_val
                X_matrix(:, col) = (X_matrix(:, col) - min_val) / (max_val - min_val);
            end
        end
        % 对特征也进行一定的 SG 平滑，去除突变噪声
        X_matrix = smoothdata(X_matrix, 'sgolay', 11);
        
        % 动态生成变量名
        suffix = curr_batt(5:end); % 提取 "35", "36" 等
        var_cycle = ['cycle_', suffix];
        var_Y = ['Y_', suffix];
        var_X = ['X_', suffix];
        
        assignin('base', var_cycle, cycles_vec);
        assignin('base', var_Y, Y_label);
        assignin('base', var_X, X_matrix);
        
        % 记录变量名以便保存
        save_vars = [save_vars, {var_cycle, var_Y, var_X}];
        
        % 绘图 (这里仍然只画容量衰减作为直观参考)
        plot(cycles_vec, Y_label, '-', 'LineWidth', 2, 'Color', colors{b}, 'DisplayName', curr_batt);
        fprintf('  ✅ %s 处理完成，共计 %d 圈。生成特征矩阵维度: %dx4\n', curr_batt, length(Y_label), size(X_matrix, 1));
    end
end

%% 4. 保存所有电池数据
if ~isempty(save_vars)
    save(fullfile(basePath, 'CALCE2_4D_processed.mat'), save_vars{:});
    fprintf('\n🎉 任务成功！已保存到 CALCE2 文件夹下的 CALCE2_4D_processed.mat\n');
end

% 美化
yline(0.8, '--k', 'EOL', 'LineWidth', 1.5);
xlabel('Cycle Count'); ylabel('Normalized Capacity (SOH)');
title('CALCE Dataset Capacity Degradation (Multi-Battery)');
legend('Location', 'northeast', 'Interpreter', 'none');