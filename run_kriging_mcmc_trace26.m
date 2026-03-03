% run_kriging_mcmc_trace26.m
% 目标：在 GaussianMixMCMC 的示例框架(main_synthetic_test.m)下，
% 1) 选定两口井：井A=Trace_005、井B=Trace_035；
% 2) 用球状变差函数的普通克里金对“目标井位 Trace_026”的弹性参数全井统计量（均值、方差）
%    与岩性转移概率矩阵 P 进行空间插值（基于两井）；
% 3) 将克里金得到的 P 作为转移先验，按 main_synthetic_test.m 的方法对 Trace_026 进行反演；
% 4) 与目标井位的真实数据进行对比，输出准确率、混淆矩阵、各类精确��/召回率/F1，以及链稳定性指标。
%
% 重要说明：
% - 本脚本沿用仓库示例的数据与前向模型设置（theta、wavelet、PRIOR_elasticLog、SNR 等）。
% - 克里金用于估计“全井统计量”(Vp/Vs/Rho 的均值与方差)与“转移矩阵 P”，
%   其中 P 将直接用于 MCMC 的马尔可夫先验；而均值/方差主要用于报告与诊断（不直接改写 PRIOR）。
% - 仅两井插值在统计上较弱，结果对变差函数参数敏感；请结合敏感性分析与更多井位加以验证。

clear; clc; close all;

%% 一、加载数据与公共设置（与 main_synthetic_test.m 保持一致）
addpath('.\functions\')
dataFile = fullfile('data','data_synth_3layers_oil_water.mat');
load(dataFile);           % 加载：real_seismic_aki, real_vp, real_vs, real_rho, real_facies, theta, wavelet, PRIOR_elasticLog, noise_mean0_std1

% 反演迭代次数
n_it = 5000;

% 选定井位索引
wellA_idx   = 5;    % 井A
wellB_idx   = 35;   % 井B
target_idx  = 26;   % 目标井位

% 横向间距（若无实际坐标，用 trace 索引映射）
dx = 25;                      % m
xA = (wellA_idx-1)*dx;
xB = (wellB_idx-1)*dx;
xT = (target_idx-1)*dx;

fprintf('井A = Trace_%03d, 井B = Trace_%03d, 目标井位 = Trace_%03d (x=%.1f m)\n', wellA_idx, wellB_idx, target_idx, xT);

%% 二、构造 1D 目标井位的反演输入（与示例一致）
real_seismic = real_seismic_aki;

% 加噪设置（与示例一致）
SNR = 10;
for ia=1:size(real_seismic,3)
    real_seismic(:,:,ia) = real_seismic(:,:,ia) + sqrt(mean(var(real_seismic(:,:,ia)))/SNR) * noise_mean0_std1(:,:,ia);
end

% 目标井位地震与真实弹性/岩相（注意与示例相同行数：1:end-1）
real_seismic1d = real_seismic(:, target_idx, :);
real_log_vp    = log(real_vp(1:end-1, target_idx));
real_log_vs    = log(real_vs(1:end-1, target_idx));
real_log_rho   = log(real_rho(1:end-1, target_idx));
real_facies_well = real_facies(1:end-1, target_idx);

I = size(real_log_vp,1);
prob_map = ones(I,1,length(PRIOR_elasticLog))/length(PRIOR_elasticLog);
SNR_par = SNR*[1 1 1 1];
PRIOR_  = PRIOR_elasticLog;

%% 三、两井 → 目标井的克里金插值（球状变差函数）
% 3.1 统计两井与目标井的“全井平均/方差”（仅用于报告比较）
statsA = compute_stats_per_well(real_vp(:,wellA_idx), real_vs(:,wellA_idx), real_rho(:,wellA_idx));
statsB = compute_stats_per_well(real_vp(:,wellB_idx), real_vs(:,wellB_idx), real_rho(:,wellB_idx));
statsT = compute_stats_per_well(real_vp(:,target_idx), real_vs(:,target_idx), real_rho(:,target_idx)); % 真实值（用于验证）

% 3.2 变差函数参数（球状）
vg.name   = 'spherical';
vg.nugget = 0.05;
vg.sill   = 1.00;
vg.range  = 800;  % m

% 3.3 两点OK权重（对所有标量通用）
locs = [xA; xB];
[w_AB, lambda_AB, k_vec, K_mat] = ok_weights_1d(locs, xT, vg);

% 3.4 六个标量：Vp_mean/Vp_var/Vs_mean/Vs_var/Rho_mean/Rho_var 的插值与克里金方差
fields = {'Vp_mean','Vp_var','Vs_mean','Vs_var','Rho_mean','Rho_var'};
valsA  = [statsA.Vp_mean, statsA.Vp_var, statsA.Vs_mean, statsA.Vs_var, statsA.Rho_mean, statsA.Rho_var];
valsB  = [statsB.Vp_mean, statsB.Vp_var, statsB.Vs_mean, statsB.Vs_var, statsB.Rho_mean, statsB.Rho_var];

pred_scalar = struct();
for i = 1:numel(fields)
    v_pred = w_AB(1)*valsA(i) + w_AB(2)*valsB(i);
    sigma_k2 = vg.sill - (w_AB.' * k_vec) - lambda_AB;   % 克里金方差（与观测值无关）
    pred_scalar.(fields{i}).value    = v_pred;
    pred_scalar.(fields{i}).var_krig = max(0, sigma_k2);
end

% 3.5 岩性转移概率矩阵：由井A/井B统计 → 对每个元素克里金 → 行归一化
classes_all = sort(unique(real_facies(:))).';
P_A = compute_transition_matrix(real_facies(:,wellA_idx), classes_all);
P_B = compute_transition_matrix(real_facies(:,wellB_idx), classes_all);

P_T_pred = zeros(size(P_A));
for ii = 1:numel(classes_all)
    for jj = 1:numel(classes_all)
        pij = w_AB(1)*P_A(ii,jj) + w_AB(2)*P_B(ii,jj);
        P_T_pred(ii,jj) = min(1,max(0,pij));
    end
    s = sum(P_T_pred(ii,:));
    if s>0, P_T_pred(ii,:) = P_T_pred(ii,:)/s; else, P_T_pred(ii,:) = ones(1,numel(classes_all))/numel(classes_all); end
end

% 目标井真实转移矩阵（用于验证）
P_T_true = compute_transition_matrix(real_facies(:,target_idx), classes_all);

fprintf('\n=== 变差函数（%s）参数 ===\n', vg.name);
fprintf('nugget=%.4f, sill=%.4f, range=%.1f m\n', vg.nugget, vg.sill, vg.range);
fprintf('OK权重：w_A=%.4f, w_B=%.4f, lambda=%.6f\n', w_AB(1), w_AB(2), lambda_AB);

fprintf('\n=== 目标井 Trace_%03d 的弹性参数“全井统计”插值与真实对比 ===\n', target_idx);
for i=1:numel(fields)
    name = fields{i};
    v_pred = pred_scalar.(name).value;
    s2k    = pred_scalar.(name).var_krig;
    v_true = statsT.(name);
    fprintf('%-10s: 预测值=%12.4f, 克里金方差=%10.6f, 真实值=%12.4f, 误差=%+12.4f\n', ...
        name, v_pred, s2k, v_true, v_pred-v_true);
end

fprintf('\n=== 目标井位 岩性转移概率矩阵（预测值-行归一化） ===\n');
disp(P_T_pred);
fprintf('=== 目标井位 岩性转移概率矩阵（真实） ===\n');
disp(P_T_true);

%% 四、将 P_T_pred 用作先验，按 main_synthetic_test.m 对 Trace_026 反演
P_for_inversion = P_T_pred;     % 用插值得到的转移矩阵替换示例中的固定P
[ INVERSION ] = GaussianMixMCMC_metropolis(real_seismic1d, theta, SNR_par, wavelet, PRIOR_, n_it, prob_map, P_for_inversion);

INVERSION_ = INVERSION;

%% 五、性能评估：准确率、混淆矩阵、精确率/召回率/F1、链稳定性
% 预测岩相（最可能序列）
pred_facies_well = INVERSION_.FACIES.likely(:);           % I x 1
true_facies_well = real_facies_well(:);                   % I x 1
K = numel(classes_all);

% 5.1 准确率与混淆矩阵
[acc, Cmat] = accuracy_and_confusion(true_facies_well, pred_facies_well, classes_all);
fprintf('\n=== 序列分类结果（Trace_%03d） ===\n', target_idx);
fprintf('总体准确率：%.4f\n', acc);
disp('混淆矩阵（行=真实，列=预测）：');
disp(array2table(Cmat, 'VariableNames', compose('Pred_%d',classes_all), 'RowNames', compose('True_%d',classes_all)));

% 5.2 每类精确率/召回率/F1
[prec, rec, f1] = precision_recall_f1(Cmat);
T_metrics = table(classes_all(:), prec(:), rec(:), f1(:), 'VariableNames', {'Class','Precision','Recall','F1'});
disp('分岩相指标：');
disp(T_metrics);

% 5.3 链稳定性（简易指标）
% 以 FACIES.samples (I x n_it) 为基础，计算相邻迭代的平均汉明距离（归一化到[0,1]）
samples = INVERSION_.FACIES.samples;  % I x n_it
if ~isempty(samples)
    d_iter = mean(samples(:,2:end) ~= samples(:,1:end-1), 1); % 每次迭代的平均变化比例
    mean_switch = mean(d_iter);
    fprintf('链稳定性：相邻迭代的平均汉明距离（全深度平均）= %.4f（越小越稳定）\n', mean_switch);
else
    fprintf('未找到 FACIES.samples，无法计算链稳定性指标。\n');
end

%% 六、可视化：弹性均值对比 + |P_pred - P_true| 热图 + 预测/真实岩相
figure('Color','w','Position',[100 100 1200 450]);

% 6.1 弹性“全井均值”对比
subplot(1,3,1);
bar([ [statsT.Vp_mean; statsT.Vs_mean; statsT.Rho_mean], ...
      [pred_scalar.Vp_mean.value; pred_scalar.Vs_mean.value; pred_scalar.Rho_mean.value] ], 0.9);
set(gca,'XTickLabel',{'Vp mean','Vs mean','Rho mean'});
legend('True','Kriging Pred','Location','northoutside'); title(sprintf('Trace %d Elastic Means', target_idx));

% 6.2 |P_pred - P_true| 热图
subplot(1,3,2);
absDiff = abs(P_T_pred - P_T_true);
imagesc(absDiff); axis image; colorbar; caxis([0 1]);
title('|P_{pred} - P_{true}|'); xlabel('To (j)'); ylabel('From (i)');

% 6.3 预测/真实岩相序列
subplot(1,3,3);
plot(1:I, true_facies_well, '-k','LineWidth',1.5); hold on;
plot(1:I, pred_facies_well, '--r','LineWidth',1.5);
legend('True','Pred','Location','northoutside');
xlabel('Sample (depth index)'); ylabel('Facies'); title('Facies Sequence @ Trace 26'); grid on;

set(gcf,'PaperPositionMode','auto');
print(gcf, 'trace26_kriging_mcmc_results.png', '-dpng', '-r300');
fprintf('结果图已保存：trace26_kriging_mcmc_results.png (300 DPI)\n');

%% 七、结论性提示（可根据需要扩展为报告）
% - 在仅两井条件下，P 的空间插值对“结构性稀疏”（如某行近零）不敏感，建议结合掩码/Top-k/Dirichlet等先验稀疏化后处理。
% - 若希望把“弹性先验”也做空间定制，应为每类（或每变量）拟合其专属 variogram，并将其融入 PRIOR 的趋势或协方差构造中。

%% ============== 辅助函数 ==============

function stats = compute_stats_per_well(Vp_col, Vs_col, Rho_col)
    stats.Vp_mean  = mean(Vp_col,'omitnan');
    stats.Vp_var   = var(Vp_col,'omitnan');
    stats.Vs_mean  = mean(Vs_col,'omitnan');
    stats.Vs_var   = var(Vs_col,'omitnan');
    stats.Rho_mean = mean(Rho_col,'omitnan');
    stats.Rho_var  = var(Rho_col,'omitnan');
end

function [w, lambda, k_vec, K] = ok_weights_1d(locs, x0, vg)
    n = numel(locs);
    K = zeros(n,n);
    for i=1:n
        for j=1:n
            K(i,j) = cov_from_variogram(abs(locs(i)-locs(j)), vg);
        end
    end
    k_vec = zeros(n,1);
    for i=1:n
        k_vec(i) = cov_from_variogram(abs(locs(i)-x0), vg);
    end
    A = [K, ones(n,1); ones(1,n), 0];
    b = [k_vec; 1];
    reg = 1e-12;
    sol = (A + reg*eye(n+1)) \ b;
    w = sol(1:n);
    lambda = sol(end);
end

function C = cov_from_variogram(h, vg)
    switch lower(vg.name)
        case 'spherical'
            if h==0
                gamma = 0;
            elseif h<vg.range
                hr = h/vg.range;
                gamma = vg.nugget + (vg.sill - vg.nugget)*(1.5*hr - 0.5*hr^3);
            else
                gamma = vg.sill;
            end
            C = max(0, vg.sill - gamma);
        case 'exponential'
            a = vg.range;
            gamma = (h==0)*0 + (h>0)*(vg.nugget + (vg.sill - vg.nugget)*(1 - exp(-h/a)));
            C = max(0, vg.sill - gamma);
        case 'gaussian'
            a = vg.range;
            gamma = (h==0)*0 + (h>0)*(vg.nugget + (vg.sill - vg.nugget)*(1 - exp(-(h/a)^2)));
            C = max(0, vg.sill - gamma);
        otherwise
            error('未知变差函数：%s', vg.name);
    end
end

function P = compute_transition_matrix(lith_seq, classes)
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
    for i=1:K
        s = sum(T(i,:));
        if s>0, P(i,:) = T(i,:)/s; else, P(i,:) = zeros(1,K); end
    end
end

function [acc, Cmat] = accuracy_and_confusion(y_true, y_pred, classes)
    K = numel(classes);
    Cmat = zeros(K,K);
    for n=1:numel(y_true)
        i = find(classes==y_true(n),1);
        j = find(classes==y_pred(n),1);
        if ~isempty(i)&&~isempty(j), Cmat(i,j)=Cmat(i,j)+1; end
    end
    acc = sum(diag(Cmat))/max(1,sum(Cmat(:)));
end

function [prec, rec, f1] = precision_recall_f1(C)
    K = size(C,1);
    prec = zeros(K,1); rec = zeros(K,1); f1 = zeros(K,1);
    for k=1:K
        tp = C(k,k);
        fp = sum(C(:,k)) - tp;
        fn = sum(C(k,:)) - tp;
        prec(k) = tp / max(1,(tp+fp));
        rec(k)  = tp / max(1,(tp+fn));
        if prec(k)+rec(k)>0
            f1(k) = 2*prec(k)*rec(k)/(prec(k)+rec(k));
        else
            f1(k) = 0;
        end
    end
end