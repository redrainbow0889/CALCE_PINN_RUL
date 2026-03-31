% ===== Fold 2：4D 架构滑动窗口动态微调与泛化评测 (完全体) =====
clear; clc; close all;

%% 1. 加载 4D 全局归一化数据与预训练模型
fprintf('加载 4D 全局测试数据与预训练模型...\n');
% 必须加载最新的 GlobalNorm 数据集！
load('D:\OneDrive\桌面\杂\毕设论文\电池充放电数据\CALCE2\CALCE2_4D_GlobalNorm_processed.mat');
load('Fold2_Pretrained_4D_Model.mat'); 

% 组装 36 号电池 (测试集) 的全局数据备用
feat_36 = [cycle_36'/1000.0; X_36(:, 2)'; X_36(:, 3)'; X_36(:, 4)']; % 4 x N
Y_36_true = Y_36';

EOL_threshold = 0.8;
pred_start_points = [150, 180, 210, 240, 270, 300]; 
results_RUL = zeros(length(pred_start_points), 3); % [真实RUL, 预测RUL, 误差]

fprintf('\n================ 开始 CS2_36 动态泛化评测 (4D 完全体) ================\n');

real_eol_idx = find(Y_36_true <= EOL_threshold, 1, 'first');
if isempty(real_eol_idx), real_eol_idx = length(Y_36_true); end
real_eol_cycle = cycle_36(real_eol_idx);

%% 2. 滑动窗口微调循环
for p = 1:length(pred_start_points)
    current_pt = pred_start_points(p);
    current_cycle = cycle_36(current_pt);
    real_RUL = real_eol_cycle - current_cycle;
    
    % --- 2.1 提取当前局部历史数据 (严格截断历史包袱，仅取最近 50 圈) ---
    window_len = 50; 
    hist_start = max(1, current_pt - window_len + 1);
    
    X_local_hist = feat_36(:, hist_start:current_pt);
    Y_local_hist = Y_36_true(hist_start:current_pt);
    
    actual_len = size(X_local_hist, 2); 
    X_hist_dl = dlarray(reshape(X_local_hist, [4, 1, actual_len]), 'CBT');
    C_hist_dl = dlarray(reshape(Y_local_hist, [1, 1, actual_len]), 'CBT');
    
    % --- 2.2 构建未来全局物理配点 (强制悲观外推) ---
    future_cycles = (current_cycle+1 : 1200);
    future_len = length(future_cycles);
    future_n_scaled = future_cycles / 1000.0;
    
    % 使用局部 50 圈拟合初始趋势
    past_n = X_local_hist(1, :)'; 
    p_vend = polyfit(past_n, X_local_hist(2, :)', 1);
    p_icpeak = polyfit(past_n, X_local_hist(3, :)', 1);
    p_icvol = polyfit(past_n, X_local_hist(4, :)', 1);
    
    % 【核心修复：物理强制悲观先验，切断乐观幻觉】
    % 如果处于平台期拟合出正斜率，强制覆写为保底衰减率 (例如每千圈下降 0.05)
    min_decay_rate = -0.05; 
    p_vend(1) = min(p_vend(1), min_decay_rate);
    p_icpeak(1) = min(p_icpeak(1), min_decay_rate);
    p_icvol(1) = min(p_icvol(1), min_decay_rate);
    
    % 计算未来特征，加入物理下界防止穿透 0
    fut_vend = max(polyval(p_vend, future_n_scaled), 0);
    fut_icpeak = max(polyval(p_icpeak, future_n_scaled), 0);
    fut_icvol = max(polyval(p_icvol, future_n_scaled), 0);
    
    X_future_extrapolated = [future_n_scaled; fut_vend; fut_icpeak; fut_icvol];
    X_pred_dl = dlarray(reshape(X_future_extrapolated, [4, 1, future_len]), 'CBT');
    
    % --- 2.3 构建无缝拼接的全时序输入 (修复 GRU 失忆症) ---
    X_all_mat = cat(3, extractdata(X_hist_dl), extractdata(X_pred_dl));
    X_all_dl = dlarray(X_all_mat, 'CBT');
    
    % --- 2.4 动态微调设置 ---
    ft_net = best_net;
    ft_k = best_k; ft_r = best_r; ft_alpha = best_alpha; ft_beta = best_beta;
    
    maxIter_FT = 600; 
    lr_fc = 0.005;    
    lr_gru = 0.0001;  
    
    tAvgNet = []; sqTAvgNet = []; tAvgK = []; sqTAvgK = [];
    tAvgR = []; sqTAvgR = []; tAvgA = []; sqTAvgA = []; tAvgB = []; sqTAvgB = [];
    
    % --- 微调迭代 ---
    for iter = 1:maxIter_FT
        [loss, gradNet, gradK, gradR, gradA, gradB] = dlfeval(@ftLoss, ...
            ft_net, ft_k, ft_r, ft_alpha, ft_beta, X_all_dl, C_hist_dl);
        
        idx_fc = contains(ft_net.Learnables.Layer, 'fc');
        for i = 1:size(gradNet, 1)
            if idx_fc(i)
                current_lr = lr_fc;
            else
                current_lr = lr_gru;
            end
            gradNet.Value{i} = gradNet.Value{i} * (current_lr / 0.001); 
        end
        
        [ft_net, tAvgNet, sqTAvgNet] = adamupdate(ft_net, gradNet, tAvgNet, sqTAvgNet, iter, 0.001);
        [ft_k, tAvgK, sqTAvgK] = adamupdate(ft_k, gradK, tAvgK, sqTAvgK, iter, lr_fc);
        [ft_r, tAvgR, sqTAvgR] = adamupdate(ft_r, gradR, tAvgR, sqTAvgR, iter, lr_fc);
        [ft_alpha, tAvgA, sqTAvgA] = adamupdate(ft_alpha, gradA, tAvgA, sqTAvgA, iter, lr_fc);
        [ft_beta, tAvgB, sqTAvgB] = adamupdate(ft_beta, gradB, tAvgB, sqTAvgB, iter, lr_fc);
        
        ft_k = max(ft_k, dlarray(1.0)); ft_alpha = max(ft_alpha, best_alpha * 0.8); ft_beta = max(ft_beta, dlarray(1.2)); 
    end
    
    % --- 2.5 预测与 RUL 计算 ---
    C_all_pred = squeeze(extractdata(predict(ft_net, X_all_dl)));
    C_pred_future = C_all_pred(actual_len + 1 : end); 
    
    window_size = 5; 
    C_pred_smoothed = movmean(C_pred_future, window_size, 'Endpoints', 'discard');
    
    pred_eol_offset = find(C_pred_smoothed <= EOL_threshold, 1, 'first');
    if isempty(pred_eol_offset)
        pred_RUL = future_cycles(end) - current_cycle; 
    else
        pred_RUL = future_cycles(pred_eol_offset + floor(window_size/2)) - current_cycle;
    end
    
    error_RUL = pred_RUL - real_RUL;
    results_RUL(p, :) = [real_RUL, pred_RUL, error_RUL];
    
    fprintf('  点 %d | 预测 RUL: %d 圈 (真实: %d 圈) | 误差: %d\n', current_pt, pred_RUL, real_RUL, error_RUL);
end

%% 3. 结果汇总
fprintf('================ 评测结束 ================\n');
MAE = mean(abs(results_RUL(:, 3)));
RMSE = sqrt(mean(results_RUL(:, 3).^2));
fprintf('当前 4D 架构平均绝对误差 (MAE): %.2f 圈\n', MAE);
fprintf('当前 4D 架构均方根误差 (RMSE): %.2f 圈\n', RMSE);

%% --- 微调专用 Loss (修复时序断裂与截距悬浮) ---
function [loss, gradNet, gradK, gradR, gradA, gradB] = ftLoss(net, k, r, alpha, beta, X_all, C_hist)
    L_hist = size(C_hist, 3);
    
    % 前向传播一整条连续时间线
    C_all_pred = forward(net, X_all);
    C_hist_pred = C_all_pred(:, :, 1:L_hist);
    C_future_pred = C_all_pred(:, :, L_hist+1:end);
    
    % 1. 历史段：指数时间加权 Loss
    idx_seq = 1:L_hist;
    weights = exp(0.1 * (idx_seq - L_hist)); 
    weights = weights / sum(weights); 
    weights_dl = dlarray(reshape(weights, [1, 1, L_hist]), 'CBT');
    data_loss = sum(weights_dl .* (C_hist_pred - C_hist).^2, 'all');
    
    % 2. 硬核锚点约束 (极其残暴地将当前最新预测点钉死在当前真实的标签上)
    anchor_loss = (C_hist_pred(1,1,end) - C_hist(1,1,end))^2;
    
    % 3. 未来段：物理残差审查
    C_phys = reshape(squeeze(stripdims(C_future_pred)), 1, []); 
    n_phy_mat = squeeze(stripdims(X_all(:, :, L_hist+1:end))); 
    n_phy = n_phy_mat(1, :); 
    
    dCdn = (C_phys(2:end) - C_phys(1:end-1)) ./ (n_phy(2:end) - n_phy(1:end-1));
    C_t = C_phys(2:end); n_t = n_phy(2:end); 
    
    physics_residual = dCdn + (r/1000.0) .* C_t .* (1 - C_t ./ k) + alpha .* exp(beta .* n_t);
    physics_loss = mean((1.0 + 10.0 * (n_t .^ 2)) .* (physics_residual.^2), 'all');
    
    % 4. 边界与单调性防线
    monotonicity_penalty = mean(max(0, dCdn).^2, 'all'); 
    bound_penalty = mean(max(0, C_future_pred - 1.0).^2 + max(0, 0.0 - C_future_pred).^2, 'all');
    
    % 给予锚点巨大的 1000.0 权重，强制拉下截距
    loss = 10.0 * data_loss + 1000.0 * anchor_loss + 5.0 * physics_loss + 100.0 * monotonicity_penalty + 100.0 * bound_penalty;
    [gradNet, gradK, gradR, gradA, gradB] = dlgradient(loss, net.Learnables, k, r, alpha, beta);
end