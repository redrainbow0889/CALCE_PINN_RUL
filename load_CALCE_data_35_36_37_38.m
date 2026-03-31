%% CALCE 数据集批量自动化提取与清洗脚本 (全量版：35, 36, 37, 38)
clear; clc;

% 1. 设置基础路径和要处理的电池编号
basePath = 'D:\OneDrive\桌面\杂\毕设论文\电池充放电数据\CALCE\'; 
battery_names = {'CS2_35', 'CS2_36', 'CS2_37', 'CS2_38'};

% 初始化图形窗口
figure('Name', 'CALCE 多电池容量衰减对比', 'Position', [100, 100, 900, 600]);
hold on; grid on;
colors = {'#0072BD', '#D95319', '#77AC30', '#7E2F8E'}; % 蓝, 橙, 绿, 紫

% 预分配一个 cell 用于最后统一保存变量名
save_vars = {};

%% 2. 开始循环处理每个电池
for b = 1:length(battery_names)
    curr_batt = battery_names{b};
    folderPath = fullfile(basePath, curr_batt);
    
    fprintf('\n🚀 正在处理: %s ...\n', curr_batt);
    
    fileList = dir(fullfile(folderPath, '*.xlsx')); 
    if isempty(fileList), continue; end
    
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

    total_capacity = []; 
    for i = 1:length(fileList)
        full_path = fullfile(fileList(i).folder, fileList(i).name);
        try
            sheets = sheetnames(full_path);
            target_sheet = sheets(contains(sheets, 'Channel', 'IgnoreCase', true));
            if isempty(target_sheet), continue; end
            
            data = readtable(full_path, 'Sheet', target_sheet(1), 'VariableNamingRule', 'preserve');
            
            % 提取放电步 (Step 7) 的电流积分
            idx = (data.("Step_Index") == 7);
            cycles = unique(data.("Cycle_Index")(idx));
            for c = 1:length(cycles)
                c_idx = (data.("Cycle_Index") == cycles(c)) & idx;
                t = data.("Test_Time(s)")(c_idx);
                I = data.("Current(A)")(c_idx);
                if length(t) > 1
                    cap = trapz(t, abs(I)) / 3600; 
                    if cap > 0.4, total_capacity = [total_capacity; cap]; end
                end
            end
        catch, continue; end
    end
    
    %% 3. 数据归一化、去噪与动态变量生成
    if ~isempty(total_capacity)
        % 归一化并使用移动中值去噪
        norm_cap = total_capacity / total_capacity(1);
        clean_cap = smoothdata(norm_cap, 'movmedian', 15)'; 
        cycles_vec = 0:(length(clean_cap)-1);
        
        % --- 关键改进：动态生成变量名 ---
        % 例如：生成变量 cycle_37 和 cap_37
        suffix = curr_batt(5:end); % 提取 "35", "36" 等
        var_cycle = ['cycle_', suffix];
        var_cap = ['cap_', suffix];
        
        assignin('base', var_cycle, cycles_vec);
        assignin('base', var_cap, clean_cap);
        
        % 记录变量名以便保存
        save_vars = [save_vars, {var_cycle, var_cap}];
        
        % 绘图
        plot(cycles_vec, clean_cap, '-', 'LineWidth', 2, 'Color', colors{b}, 'DisplayName', curr_batt);
        fprintf('  ✅ %s 处理完成，共计 %d 圈。\n', curr_batt, length(clean_cap));
    end
end

%% 4. 保存所有电池数据
if ~isempty(save_vars)
    save('CALCE_processed.mat', save_vars{:});
    fprintf('\n🎉 任务成功！已保存 [%s] 到 CALCE_processed.mat\n', strjoin(save_vars, ', '));
end

% 美化
yline(0.8, '--k', 'EOL', 'LineWidth', 1.5);
xlabel('Cycle Count'); ylabel('Normalized Capacity (SOH)');
title('CALCE Dataset Capacity Degradation (Multi-Battery)');
legend('Location', 'northeast', 'Interpreter', 'none');