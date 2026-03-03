%% 变换马尔科夫链转移矩阵P
function [P_upward] = transpose_trasition_matrix(P)  % 输入为转移矩阵P，输出为变换后的矩阵P_upward

n_facies = size(P,1);  % 岩性类别数量（即矩阵的行数）

prop = P^100;
prop = prop(1,:);  % prop取的是P乘方之后的第一行，表示经过许多步转移后，各岩性的“平稳分布”概率

P_upward = P';  % 对原矩阵取转置，得到“上行”转移矩阵
P_upward = P_upward.*repmat(prop,n_facies,1);  % 用prop做行归一化处理，使每行的概率总和为1
P_upward = P_upward./repmat(sum(P_upward,2),1,n_facies);  % 矩阵复制，方便做元素级运算

if sum(sum(isnan(P_upward))) >0
    P_upward = ones(n_facies,n_facies)/n_facies;
    %P_upward = P_upward./repmat(sum(P_upward,2),1,n_facies);
end