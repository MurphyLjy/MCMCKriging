% kriging_predict_target_well.m
% 基于 data_synth_3layers_oil_water.mat：
% - 选取两口井（井A、井B）与其间随机目标井位（trace）
% - 使用球状变差函数的普通克里金（OK）插值：
%   * 六个标量：Vp均值/方差、Vs均值/方差、Rho均值/方差
%   * 岩性转移概率矩阵（每元素）
% - 在目标井位用真实数据进行验证（输出误差与对比）
%
% 说明：
% - 数据集未提供显式XY坐标；脚本将横向trace索引映射为距离坐标 x = (trace-1)*dx
% - 可将井A/井B的trace索引手动设定；目标井位默认在二者之间随机选择
% - 变差函数模型默认使用球状（spherical）模型；可切换为指数/高斯等

clear; clc; close all;

%% 0) 参数与井位选择
dataFile = fullfile('data','data_synth_3layers_oil_water.mat');
if ~isfile(dataFile)
    error('未找到数据文件：%s。请确认路径。', dataFile);
end
S = load(dataFile);

% 必需变量检查
reqVars = {'real_vp','real_vs','real_rho','real_facies'};
for v = 1:numel(reqVars)
    if ~isfield(S, reqVars{v})
        error('数据文件中缺少必需变量：%s', reqVars{v});
    end
end

Vp2D = S.real_vp;   % [depth x trace]
Vs2D = S.real_vs;   % [depth x trace]
Rho2D= S.real_rho;  % [depth x trace]
fac2D= S.real_facies; % [depth x trace]
[nSamples, nTraces] = size(Vp2D);

% 设定井A/井B（trace 索引），可按需修改
wellA_idx = 5; 
wellB_idx = 35;
if wellA_idx<1 || wellA_idx>nTraces || wellB_idx<1 || wellB_idx>nTraces
    error('井A或井B索引越界：nTraces=%d，wellA_idx=%d，wellB_idx=%d', nTraces, wellA_idx, wellB_idx);
end

% 目标井位：在A与B之间随机选择
rng(123); % 固定随机种子以便复现
minIdx = min(wellA_idx, wellB_idx)+1;
maxIdx = max(wellA_idx, wellB_idx)-1;
if minIdx>maxIdx
    % 如果A与B相邻，则选择其中点向右一列（若存在）
    target_idx = min(max(wellA_idx,wellB_idx)+1, nTraces);
else
    target_idx = randi([minIdx, maxIdx]);
end

fprintf('井A = Trace_%03d, 井B = Trace_%03d, 目标井位 = Trace_%03d\n', wellA_idx, wellB_idx, target_idx);

% 将trace索引映射为横向坐标（米）
dx = 25; % 假定横向采样间距；可根据实际数据/项目设定
xA = (wellA_idx-1)*dx;
xB = (wellB_idx-1)*dx;
xT = (target_idx-1)*dx;

%% 1) 变差函数模型（球状）参数设定
% spherical variogram: 
% γ(h) = nugget + (sill - nugget)*(1.5*(h/r) - 0.5*(h/r)^3), for 0<h<r; γ(h)=sill for h>=r; γ(0)=0
% 协方差：C(h) = sill - γ(h)；C(0)=sill
variogramModel.name   = 'spherical';
variogramModel.nugget = 0.05;   % 块金值（可根据区域经验调整）
variogramModel.sill   = 1.00;   % 总台（总方差）
variogramModel.range  = 800;    % 变程（米），影响空间相关尺度

% 若需其他模型，可在 cov_from_variogram 内添加 'exponential'/'gaussian' 分支，并调整参数：
% variogramModel.name   = 'exponential'; variogramModel.nugget=...; variogramModel.sill=...; variogramModel.range=... 

%% 2) 计算井A/井B/目标井的六个标量（样本均值/方差）
% 注意：这里的均值/方差是“样本统计量”，用来作为区域化变量进行空间插值；
% 插值输出的“克里金方差”是插值本身的不确定性，不是样本方差。
statsA = compute_stats_per_well(Vp2D(:,wellA_idx), Vs2D(:,wellA_idx), Rho2D(:,wellA_idx));
statsB = compute_stats_per_well(Vp2D(:,wellB_idx), Vs2D(:,wellB_idx), Rho2D(:,wellB_idx));
statsT_true = compute_stats_per_well(Vp2D(:,target_idx), Vs2D(:,target_idx), Rho2D(:,target_idx)); % 用于验证

% 将六个量放入结构数组便于循环插值
fields = {'Vp_mean','Vp_var','Vs_mean','Vs_var','Rho_mean','Rho_var'};
valsA = [statsA.Vp_mean, statsA.Vp_var, statsA.Vs_mean, statsA.Vs_var, statsA.Rho_mean, statsA.Rho_var];
valsB = [statsB.Vp_mean, statsB.Vp_var, statsB.Vs_mean, statsB.Vs_var, statsB.Rho_mean, statsB.Rho_var];

%% 3) 两点普通克里金插值（六个标量）
% 对于仅有两口井的情形，OK 权重只取决于距离与变差函数，因此六个变量共享同一组权重；
% 我们先建立位置与协方差系统，然后对每个变量的值应用该权重。
locs = [xA; xB];               % 1D 横向坐标
vals_placeholder = [1; 2];     % 占位（求权重时不用真实值）
[w_AB, lambda_AB, k_vec, K_mat] = ok_weights_1d(locs, xT, variogramModel);

% 逐变量插值与克里金方差
pred_scalar = struct();
for i = 1:numel(fields)
    vA = valsA(i); vB = valsB(i);
    v_pred = w_AB(1)*vA + w_AB(2)*vB;
    % 克里金方差：sigma_k^2 = C(0) - w'k - lambda
    sigma_k2 = variogramModel.sill - (w_AB.' * k_vec) - lambda_AB;
    pred_scalar.(fields{i}).value = v_pred;
    pred_scalar.(fields{i}).var_krig = max(0, sigma_k2); % 数值截断到非负
end

%% 4) 计算井A/井B的岩性转移概率矩阵，并对矩阵元素做克里金插值
classes_all = sort(unique(fac2D(:))).'; % 所有出现过的岩性编码
P_A = compute_transition_matrix(fac2D(:,wellA_idx), classes_all);
P_B = compute_transition_matrix(fac2D(:,wellB_idx), classes_all);

% 对每个元素插值；随后进行行归一化
P_T_pred = zeros(size(P_A));
for ii = 1:numel(classes_all)
    for jj = 1:numel(classes_all)
        pij_A = P_A(ii,jj);
        pij_B = P_B(ii,jj);
        pij_pred = w_AB(1)*pij_A + w_AB(2)*pij_B;
        % 概率截断到 [0,1]
        pij_pred = max(0, min(1, pij_pred));
        P_T_pred(ii,jj) = pij_pred;
    end
end
% 行归一化（每个源岩性行和为1）
for ii = 1:numel(classes_all)
    rowSum = sum(P_T_pred(ii,:));
    if rowSum>0
        P_T_pred(ii,:) = P_T_pred(ii,:) / rowSum;
    else
        % 若该源岩性在两井均未发生转移，退化为均匀分布
        P_T_pred(ii,:) = ones(1,numel(classes_all)) / numel(classes_all);
    end
end

%% 5) 目标井位真实转移矩阵（用于验证）
P_T_true = compute_transition_matrix(fac2D(:,target_idx), classes_all);

%% 6) 输出结果与验证
fprintf('\n=== 变差函数模型参数（%s）===\n', variogramModel.name);
fprintf('nugget = %.4f, sill = %.4f, range = %.1f m\n', variogramModel.nugget, variogramModel.sill, variogramModel.range);

fprintf('\n=== 克里金权重（对六个标量通用）===\n');
fprintf('w_A = %.4f, w_B = %.4f, lambda = %.6f\n', w_AB(1), w_AB(2), lambda_AB);

fprintf('\n=== 目标井位（Trace_%03d, x=%.1f m）弹性参数插值结果与验证 ===\n', target_idx, xT);
for i = 1:numel(fields)
    name = fields{i};
    v_pred = pred_scalar.(name).value;
    s2_k   = pred_scalar.(name).var_krig;
    % 真实统计值（目标井）
    v_true = statsT_true.(name);
    fprintf('%-10s: 预测值 = %10.4f, 克里金方差 = %10.6f, 真实值 = %10.4f, 误差(预测-真实) = %+10.4f\n',...
        name, v_pred, s2_k, v_true, v_pred - v_true);
end

fprintf('\n=== 目标井位岩性转移概率矩阵预测值（已行归一化） ===\n');
disp(P_T_pred);

fprintf('=== 目标井位真实岩性转移概率矩阵（用于验证） ===\n');
disp(P_T_true);

% 简单误差度量：元素级绝对差的均值
absDiff = abs(P_T_pred - P_T_true);
fprintf('转移矩阵元素绝对差的平均值 = %.6f\n', mean(absDiff(:)));

%% 7) 可选：可视化对比（弹性参数与转移矩阵）
figure('Color','w','Position',[100 100 1100 400]);
subplot(1,2,1);
bar([ [statsT_true.Vp_mean; statsT_true.Vs_mean; statsT_true.Rho_mean], ...
      [pred_scalar.Vp_mean.value; pred_scalar.Vs_mean.value; pred_scalar.Rho_mean.value] ]);
set(gca,'XTickLabel',{'Vp mean','Vs mean','Rho mean'});
legend('True','Kriging Pred'); title(sprintf('Trace %d Elastic Means', target_idx));

subplot(1,2,2);
imagesc(absDiff); colorbar; title('|P_{pred} - P_{true}|');
xlabel('To (j)'); ylabel('From (i)');

%% ======= 函数定义 =======

function stats = compute_stats_per_well(Vp_col, Vs_col, Rho_col)
% 计算单井的样本均值与样本方差（忽略 NaN）
    stats.Vp_mean  = mean(Vp_col,'omitnan');
    stats.Vp_var   = var(Vp_col,'omitnan'); % 样本方差
    stats.Vs_mean  = mean(Vs_col,'omitnan');
    stats.Vs_var   = var(Vs_col,'omitnan');
    stats.Rho_mean = mean(Rho_col,'omitnan');
    stats.Rho_var  = var(Rho_col,'omitnan');
end

function [w, lambda, k_vec, K] = ok_weights_1d(locs, x0, vg)
% 两点普通克里金权重（可扩展到n点）；使用球状/指数/高斯等变差函数转换为协方差
% locs: n x 1 采样点坐标（米）
% x0  : 目标点坐标
% vg  : 变差函数参数 struct（name,nugget,sill,range）

    n = numel(locs);
    K = zeros(n,n);
    for i = 1:n
        for j = 1:n
            h = abs(locs(i) - locs(j));
            K(i,j) = cov_from_variogram(h, vg);
        end
    end
    k_vec = zeros(n,1);
    for i = 1:n
        h0 = abs(locs(i) - x0);
        k_vec(i) = cov_from_variogram(h0, vg);
    end
    % OK 线性系统：[K 1; 1' 0] [w; lambda] = [k; 1]
    A = [K, ones(n,1); ones(1,n), 0];
    b = [k_vec; 1];
    % 数值正则（避免奇异）
    reg = 1e-12;
    sol = (A + reg*eye(n+1)) \ b;
    w = sol(1:n);
    lambda = sol(end);
end

function C = cov_from_variogram(h, vg)
% 根据变差函数模型计算协方差：C(h) = sill - γ(h)，且 C(0)=sill
    switch lower(vg.name)
        case 'spherical'
            if h==0
                gamma = 0;
            elseif h < vg.range
                hr = h / vg.range;
                gamma = vg.nugget + (vg.sill - vg.nugget) * (1.5*hr - 0.5*hr^3);
            else
                gamma = vg.sill; % h>=range 时 γ(h)=sill
            end
            C = max(0, vg.sill - gamma);
        case 'exponential'
            % γ(h) = nugget + (sill - nugget)*(1 - exp(-h/a))
            a = vg.range;
            if h==0
                gamma = 0;
            else
                gamma = vg.nugget + (vg.sill - vg.nugget)*(1 - exp(-h/a));
            end
            C = max(0, vg.sill - gamma);
        case 'gaussian'
            % γ(h) = nugget + (sill - nugget)*(1 - exp(-(h/a)^2))
            a = vg.range;
            if h==0
                gamma = 0;
            else
                gamma = vg.nugget + (vg.sill - vg.nugget)*(1 - exp(-(h/a)^2));
            end
            C = max(0, vg.sill - gamma);
        otherwise
            error('不支持的变差函数模型：%s', vg.name);
    end
end

function P = compute_transition_matrix(lith_seq, classes)
% 从垂向序列计算一阶转移概率矩阵（行=当前岩性i，列=下一层岩性j）
    K = numel(classes);
    T = zeros(K,K);
    valid = ~isnan(lith_seq);
    for t = 1:numel(lith_seq)-1
        if valid(t) && valid(t+1)
            i = find(classes==lith_seq(t), 1);
            j = find(classes==lith_seq(t+1), 1);
            if ~isempty(i) && ~isempty(j)
                T(i,j) = T(i,j) + 1;
            end
        end
    end
    P = zeros(K,K);
    for i = 1:K
        s = sum(T(i,:));
        if s>0
            P(i,:) = T(i,:) / s;
        else
            P(i,:) = zeros(1,K);
        end
    end
end