%% 已知起点和终点的言行类别，按照给定的转移概率矩阵P和概率修正prob_map，模拟生成长度为n的岩性序列（马尔科夫链），可模拟多条nsims序列
% P：转移概率矩阵（描述岩性从一种转到另一种的概率）
% n：模拟序列长度
% initial_facies：初始岩性类别（数值索引）
% final_facies：最后一个位置的岩性类别（数值索引）
% nsims：要生成的序列条数（默认1）
% prob_map：每个点属于不同岩性的概率修正（如后验概率信息）
function [simulacao]  = simulate_markov_chain_finalDefined(P, n, initial_facies, final_facies, nsims, prob_map)

% 若没指定nsims，默认只模拟一条序列
if nargin<4
    nsims=1;
end

% 计算岩性类别数，调用上行转移矩阵(P_T),初始化输出数组，并设置序列首尾岩性
n_facies = size(P,2);

P_T = transpose_trasition_matrix(P);

simulacao = zeros(n,nsims);

simulacao(1,:) = initial_facies;
simulacao(end,:) = final_facies;

%% 情况1：没有概率修正
% 只用转移矩阵P，按马尔可夫链规则依次采样中间位置的岩性
if nargin<5
    for j=1:nsims
        for i = 2:n-1   
        facies = find( rand < cumsum(P(simulacao(i-1,j),:)));
        simulacao(i,j) = facies(1);    
        end
    end
%% 情况2：有概率修正
% 前半部分(2:n-2)：每一步的采样概率由当前状态的转移概率和prob_map共同决定
% 最后一个中间点(i=n-1)：同时考虑前一个位置的转移概率、后一个已知终点的“逆向转移概率”(P_T)，以及prob_map
% 所有概率归一化后，使用累积概率法采用新言行
else
   for j=1:nsims
       for i = 2:n-2
       probabilities = P(simulacao(i-1,j),:).*reshape(prob_map(i,1,:),1,n_facies);
       probabilities = probabilities./sum(probabilities);
       facies = find( rand < cumsum( probabilities ));
       simulacao(i,j) = facies(1);    
       end
       i=n-1;
       probabilities = P(simulacao(i-1,j),:).*P_T(simulacao(i+1,j),:).*reshape(prob_map(i,1,:),1,n_facies);
       probabilities = probabilities./sum(probabilities);
       facies = find( rand < cumsum( probabilities ));
       simulacao(i,j) = facies(1);    
   end
end
