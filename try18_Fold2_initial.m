% ===== Fold 2：动态微调与泛化评测 (Test on CS2_36) =====
clear; clc; close all;

%% 1. 加载测试电池数据 (CS2_36)
fprintf('加载测试集数据 CS2_36...\n');
load('CALCE_processed.mat'); 
hi36 = load('CALCE_CS2_36_HI.mat');

HI_36 = interp1(hi36.all_cycles, hi36.clean_HI_DECVD, cycle_36, 'linear', 'extrap');
HI_36_norm = HI_36/max(HI_36);
X_all_36 = [cycle_36/1000.0; HI_36_norm];

X0_dl = dlarray(reshape([0; 1.0], [2, 1, 1]), 'CBT');
C0_dl = dlarray(reshape(1.0, [1, 1, 1]), 'CBT');

%% 2. 加载 Fold 2 预训练大脑
fprintf('加载预训练全局大脑...\n');
if ~isfile('Fold2_Pretrained_Model.mat')
    error('❌ 找不到 Fold2_Pretrained_Model.mat，请先运行 try21！');
end
load('Fold2_Pretrained_Model.mat');
fprintf('✅ 成功加载预训练模型！(Seed: %d, RMSE: %.5f)\n', best_seed, best_train_rmse);

%% 3. Fold 2 动态评测阶段 (Test on 36)
fprintf('\n================ 开始 CS2_36 动态泛化评测 (全面解冻) ================\n');
% CS2_36 的寿命较短，调整预测锚点
start_pts = [150, 180, 210, 240, 270, 300]; 
EOL_threshold = 0.8;

true_EOL_idx = find(cap_36 <= EOL_threshold, 1, 'first');
true_EOL = cycle_36(true_EOL_idx);
true_RULs = true_EOL - start_pts;
pred_RULs = zeros(size(start_pts));

for i = 1:length(start_pts)
    current_pt = start_pts(i);
    
    X_ft = X_all_36(:, 1:current_pt);
    C_ft = cap_36(1:current_pt);
    X_data_dl_ft = dlarray(reshape(X_ft, [2, 1, current_pt]), 'CBT');
    C_data_dl_ft = dlarray(reshape(C_ft, [1, 1, current_pt]), 'CBT');
    
    % 重置微调状态为预训练的最优状态
    ft_net = best_net; ft_k = best_k; ft_r = best_r; ft_alpha = best_alpha; ft_beta = best_beta;
    tAvgNet = []; sqTAvgNet = []; tAvgK = []; sqTAvgK = [];
    tAvgR = []; sqTAvgR = []; tAvgA = []; sqTAvgA = []; tAvgB = []; sqTAvgB = [];
    
    % 维持双速微调策略
   maxIter_FT = 250;  % 迭代次数减半
   lr_net = 0.0005;   % 进一步压低网络学习率，死守 GRU 的时序特征
   lr_phys = 0.002;   % 物理参数适度放开，但不允许暴走
    
  for iter = 1:maxIter_FT 
        [loss_ft, gradNet, gradK, gradR, gradA, gradB] = dlfeval(@modelLossSingle, ...
            ft_net, ft_k, ft_r, ft_alpha, ft_beta, X_data_dl_ft, C_data_dl_ft, X0_dl, C0_dl, 5.0);
        
        % ================= [核心修复 1] =================
        % 拦截 GRU 梯度！只允许 FC 层微调，保护预训练的时序提取能力
        idx_fc = contains(ft_net.Learnables.Layer, 'fc'); 
        gradNet.Value(~idx_fc) = cellfun(@(x) zeros(size(x), 'like', x), gradNet.Value(~idx_fc), 'UniformOutput', false);
        % ================================================

        [ft_net, tAvgNet, sqTAvgNet] = adamupdate(ft_net, gradNet, tAvgNet, sqTAvgNet, iter, lr_net);
        [ft_k, tAvgK, sqTAvgK] = adamupdate(ft_k, gradK, tAvgK, sqTAvgK, iter, lr_phys);
        [ft_r, tAvgR, sqTAvgR] = adamupdate(ft_r, gradR, tAvgR, sqTAvgR, iter, lr_phys);
        [ft_alpha, tAvgA, sqTAvgA] = adamupdate(ft_alpha, gradA, tAvgA, sqTAvgA, iter, lr_phys);
        [ft_beta, tAvgB, sqTAvgB] = adamupdate(ft_beta, gradB, tAvgB, sqTAvgB, iter, lr_phys);
        
        % 防止物理加速项消失
        ft_alpha = max(ft_alpha, best_alpha * 0.5); 
        ft_beta  = max(ft_beta, best_beta * 0.8);   
        ft_k = max(ft_k, dlarray(1.0));
    end
    
    % 折中版宏观特征诚实外推
% ================= [优化：动态曲率的锚点平滑投影] =================
    current_hi_val = HI_36_norm(current_pt);
    current_n_scaled = cycle_36(current_pt) / 1000.0;
    
    idx_start = max(1, current_pt - 30);
    recent_n = cycle_36(idx_start : current_pt) / 1000.0;
    recent_HI = HI_36_norm(idx_start : current_pt);
    
    p_local = polyfit(recent_n, recent_HI, 1);
    local_slope = p_local(1);
    
    % 基础物理约束：强制衰退
    slope = max(min(local_slope, -0.2), -2.0); 
    
    future_n_scaled = cycle_36(current_pt+1 : end) / 1000.0;
    delta_n = future_n_scaled - current_n_scaled;
    
    % [核心优化]：引入随时间指数增加的动态加速度
    % 随着 current_n_scaled 的变大，衰退曲率会自动成倍增加，完全依靠数学内驱力
    dynamic_acc = 0.5 * exp(2.0 * current_n_scaled); 
    
    % 从当前点出发，应用动态加速度进行抛物线外推
    future_HI_pred = current_hi_val + slope * delta_n - dynamic_acc * (delta_n.^2);
    future_HI_pred = max(future_HI_pred, 0.01); % 物理兜底
    % ==============================================================
    honest_HI_norm = [reshape(HI_36_norm(1:current_pt), 1, []), reshape(future_HI_pred, 1, [])];
    X_pred_dl = dlarray(reshape([reshape(cycle_36,1,[])/1000.0; honest_HI_norm], [2, 1, length(cycle_36)]), 'CBT');
    
    % 预测与评估
    C_pred_all = squeeze(extractdata(predict(ft_net, X_pred_dl)));
    pred_EOL_idx = find(C_pred_all <= EOL_threshold, 1, 'first');
    if isempty(pred_EOL_idx)
        current_pred_EOL = max(cycle_36);
    else
        current_pred_EOL = cycle_36(pred_EOL_idx);
    end
    
    pred_RULs(i) = current_pred_EOL - current_pt;
    fprintf('  点 %d | 预测 RUL: %d 圈 (真实: %d 圈) | 误差: %d\n', current_pt, pred_RULs(i), true_RULs(i), abs(pred_RULs(i)-true_RULs(i)));
end

%% 4. 绘图
figure('Name', 'Fold 2: Test on CS2_36', 'Position', [100, 100, 600, 500]);
plot(start_pts, true_RULs, 'k--', 'LineWidth', 2, 'DisplayName', '真实 RUL'); hold on; grid on;
fill([start_pts, fliplr(start_pts)], [true_RULs.*1.1, fliplr(true_RULs.*0.9)], ...
    [0.98, 0.90, 0.71], 'FaceAlpha', 0.6, 'EdgeColor', 'none', 'DisplayName', '\pm10% 精度锥');
plot(start_pts, pred_RULs, 'ro-', 'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', '泛化测试预测 RUL');
xlabel('预测起始点 (\lambda)', 'FontWeight', 'bold'); ylabel('剩余寿命 RUL');
title('Cross-Validation Fold 2 (Test on CS2\_36)', 'FontSize', 12);
legend;

%% --- 辅助函数 ---
function [loss, gradNet, gradK, gradR, gradA, gradB] = modelLossSingle(net, k, r, a, b, X, C, X0, C0, lambda)
    loss = calcSingleLoss(net, k, r, a, b, X, C, X0, C0, lambda);
    [gradNet, gradK, gradR, gradA, gradB] = dlgradient(loss, net.Learnables, k, r, a, b);
end

function loss = calcSingleLoss(net, k, r, alpha, beta, X_data, C_data, X0, C0, lambda_physics)
    initial_loss = mse(forward(net, X0), C0); 
    C_pred = forward(net, X_data);
    data_loss = mse(C_pred, C_data);
    C_phys = reshape(squeeze(stripdims(C_pred)), 1, []); 
    n_phy = squeeze(stripdims(X_data)); n_phy = n_phy(1, :); 
    dCdn = (C_phys(2:end) - C_phys(1:end-1)) ./ (n_phy(2:end) - n_phy(1:end-1));
    C_t = C_phys(2:end); n_t = n_phy(2:end); 
    physics_residual = dCdn + (r/1000.0) .* C_t .* (1 - C_t ./ k) + alpha .* exp(beta .* n_t);
    physics_loss = mean((1.0 + 10.0 * (n_t .^ 2)) .* (physics_residual.^2), 'all');
    penalty_loss = mean(max(0, C_pred - 1.0).^2, 'all');
    loss = data_loss + lambda_physics * physics_loss + 1.0 * initial_loss + 100.0 * penalty_loss;
end