%% 根据给定的马尔可夫转移矩阵P，从指定的初始岩性出发，生成长度为n的一维岩性序列（马尔科夫链），可以同时生成多条nsims序列。支持可选的点位先验概率prob_map，用于结合贝叶斯推断结果修正采样概率。
% 输出大小为(n × nsims)的矩阵，每一列是一条模拟出的岩性序列  
function [simulation]  = simulate_markov_chain(P, n, initial_facies, nsims, prob_map)
% output:   simulation - 1D facies simulations (n X nsims)-matrix, each column is a 1D simulation
% input:    P - Transition matrix
%           n - Size of the simulation
%           initial_facies - Initial facies of the chain to start the simulation
%           nsims - Number of simulations
%           prob_map (optional) - Pointwise prior probability, usually comes from the Bayesian inference/classification

% 没有输入nsims，默认只模拟一条序列
if nargin<4
    nsims=1;
end

% 计算岩性类别数，初始化输出矩阵，并设置所有序列的初始岩性
n_facies = size(P,2);

simulation = zeros(n,nsims);

simulation(1,:) = initial_facies;

%% 不带先验概率
% 只用转移概率矩阵P，按照采样每个位置的岩性
% rand < cumsum(P(...)):按照概率分布随机采样
if nargin<5
    for j=1:nsims
        for i = 2:n   
        facies = find( rand < cumsum(P(simulation(i-1,j),:)));
        simulation(i,j) = facies(1);    
        end
    end
    
%% 带先验概率
% 每一步的采样概率由上一点的转移概率P与当前位置的先验概率prob_map相乘，并归一化后采样
else
   for j=1:nsims
       for i = 2:n   
       probabilities = P(simulation(i-1,j),:).*reshape(prob_map(i,1,:),1,n_facies );
       probabilities = probabilities./sum(probabilities);
       facies = find( rand < cumsum( probabilities ));
       simulation(i,j) = facies(1);    
       end
   end
end