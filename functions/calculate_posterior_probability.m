%% 根据大量采样结果（均值MUsamples和方差Csamples），统计每个位置上的物理参数的后验概率分布，并找出最可能的取值（最大概率点）
% MUsamples：采样得到的均值矩阵（每一列是一次采样，每一行为一个空间点）
% Csamples：采样得到的方差矩阵（结构与MUsamples相同）
% upscale：上采样倍数（决定是否对剖面插值细化，常取1）
% axis：可选，概率分布的横坐标（即参数可能的取值范围，不传则自动生成）

% probability：每个空间点在每个axis取值上的概率密度（二维矩阵）
% axis：概率分布横坐标（参数取值范围）
% most_likely：每个空间点概率最大的参数值（后验最大概率点）
function [probability,axis,most_likely] = calculate_posterior_probability(MUsamples,Csamples,upscale,axis)
    
    % 如果没有用户指定axis，则根据所有采样均值±标准差自动生成一个区间，划分成60个点作为概率分布横坐标
    if nargin<4
        axis = linspace(min(MUsamples(:)-sqrt(Csamples(:))),max(MUsamples(:)+sqrt(Csamples(:))),60)';
    end
    % 构建概率统计矩阵，行数等于剖面点数（可上采样），列数等于axis的长度
    probability = zeros(upscale*size(MUsamples,1),length(axis));
    
    %% 主概率统计循环
    % 需要上采样
    % 对每次采样，利用插值函数imresize将剖面加密，然后按高斯分布normpdf统计每个点在每个axis取值上的概率密度
    % 将所有采样的概率密度累加
    if upscale>1
        for sample = 1:size(MUsamples,2)       
            MUsamples_ = imresize( MUsamples(:,sample),[upscale*size(MUsamples(:,sample),1),size(MUsamples(:,sample),2)]);
            Csamples_ = imresize( Csamples(:,sample),[upscale*size(Csamples(:,sample),1),size(Csamples(:,sample),2)]);
        
            probability = probability + normpdf(axis',MUsamples_,sqrt(Csamples_));
        end
    
    % 不需要上采样
    % 直接用原始剖面长度，统计概率密度并累加
    else
    for sample = 1:size(MUsamples,2)       
        probability = probability + normpdf(axis',MUsamples(:,sample),sqrt(Csamples(:,sample)));
    end
    end
    %% 找出最大概率点
    % 对每个空间点，找出概率最大的axis位置，作为该点最可能的物理参数值
    [~,indexes] = max(probability');
    most_likely = axis(indexes)';
    
end