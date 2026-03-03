%% 构造指数型空间协方差矩阵，描述地质参数在不同空间位置上的相关性
% sgm2：每个点的方差（可为向量或标量，向量表示每个点的方差，标量表示所有点方差相同
% L：协方差的相关距离，L越大，相关性衰减越慢
% order：协方差衰减的阶数，1-指数型，2-高斯型
function [covar] = covariance_matrix_exp(sgm2,L,order)

% 取输入方差的平方根，得到标准差sgm
% I是点的数量
% meshgrid生成二维网格坐标，后续计算任意两点间的距离
sgm = sqrt(sgm2);
I = length(sgm);

[X,Y] = meshgrid([1:I],[1:I]);

% 计算每一对点之间的距离差abs(X(:)-Y(:)
% 用指数型公式exp(- (距离 / L )^order )计算相关性，距离越近相关性越高；order=1时为普通指数型，order=2时为高斯型
% 结果reshape成|x|的协方差矩阵
covar = exp(-(abs(X(:)-Y(:))/L).^order);
covar = reshape(covar,I,I);

% SGM：标准差的对角阵
SGM = diag(sqrt(sgm2));

% 两边左乘右乘，实现不同点方差的加权，最终得到每对点的真实协方差
covar = SGM *covar * SGM ;
