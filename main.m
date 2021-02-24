clear; clc; close all;

s = rng('default');

%% Parameters

d          = 10;   % problem dimention
batch_size = 200;   % batch size
nodes_num  = 20;   % number of agents in the network
K          = batch_size * nodes_num; % number of data points
radius     =  0.5;



iter_num=1000;
sq_grad_DGD_FIXED_a = zeros(iter_num, 1);
xminuxbar_DGD_FIXED_a = zeros(iter_num, 1);
time_DGD_FIXED_a = zeros(iter_num, 1);


sq_grad_DGD_DIMINISHING_a = zeros(iter_num, 1);
xminuxbar_DGD_DIMINISHING_a = zeros(iter_num, 1);
time_DGD_DIMINISHING_a = zeros(iter_num, 1);


sq_grad_DFO_PDA_a = zeros(iter_num, 1);
xminuxbar_DFO_PDA_a = zeros(iter_num, 1);
time_DFO_PDA_a = zeros(iter_num, 1);



sq_grad_DDZO_PDA_a = zeros(iter_num, 1);
xminuxbar_DDZO_PDA_a = zeros(iter_num, 1);
time_DDZO_PDA_a = zeros(iter_num, 1);



sq_grad_DFO_DGT_a = zeros(iter_num, 1);
xminuxbar_DFO_DGT_a = zeros(iter_num, 1);
time_DFO_DGT_a = zeros(iter_num, 1);


sq_grad_DDZO_DGT_a = zeros(iter_num, 1);
xminuxbar_DDZO_DGT_a = zeros(iter_num, 1);
time_DDZO_DGT_a = zeros(iter_num, 1);


sq_grad_xFILTER_a = zeros(iter_num, 1);
xminuxbar_xFILTER_a = zeros(iter_num, 1);
time_xFILTER_a = zeros(iter_num, 1);


sq_grad_GPDA_a = zeros(iter_num, 1);
xminuxbar_GPDA_a = zeros(iter_num, 1);
time_GPDA_a = zeros(iter_num, 1);


sq_grad_Prox_GPDA_a = zeros(iter_num, 1);
xminuxbar_Prox_GPDA_a = zeros(iter_num, 1);
time_Prox_GPDA_a = zeros(iter_num, 1);
avera=20;
for ap=1:avera
% Gradient function

gc = @(x, lambda, alpha, z, y, bs, M) 1/(bs*M)*(-y * z)/(1+exp(y*x.'*z))+1/(M)*((2*lambda*alpha*x)./((1+alpha*x.^2).^2));

% Cost function

cf = @(x, lambda, alpha, z, y, bs, M) 1/(bs*M)*(log(1+exp(-y*x.'*z))) + 1/(M)* ((lambda*alpha*x.^2)./((1+alpha*x.^2)));
 
function_lambda = 0.001;
function_aalpha = 1;

%% Data
features = randn(d,K);
labels= randi([1,2], 1, K); labels(labels==2) = -1; % labels \in {-1,1}
features_norm = features/norm(features,'fro');
big_L=1/(batch_size)*norm(features_norm,'fro')^2+2*function_lambda*function_aalpha*d;

[Adj, degree, num_of_edge,A,B,D,Lm,edge_index, eig_Lm,min_eig_Lm,WW,LN,L_hat,eig_L_hat,min_eig_L_hat] = Generate_Graph(nodes_num,radius,d);

y_temp = zeros(nodes_num*d,iter_num);
y_temp(:,1) = randn(nodes_num*d,1);

[W_small, L_small] = transform_graph(Adj, nodes_num);

W_aug = sparse(kron(W_small, eye(d)));
L_aug = sparse(kron(L_small, eye(d)));




[xminuxbar_DGD_FIXED, sq_grad_DGD_FIXED, time_DGD_FIXED] = DGD_FIXED(W_aug, y_temp, d,nodes_num,gc,function_lambda,function_aalpha, features, labels,batch_size,iter_num);
[xminuxbar_DGD_DIMINISHING, sq_grad_DGD_DIMINISHING, time_DGD_DIMINISHING] = DGD_DIMINISHING(W_aug, y_temp, d,nodes_num,gc,function_lambda,function_aalpha, features, labels,batch_size,iter_num);
[xminuxbar_DFO_PDA, sq_grad_DFO_PDA, time_DFO_PDA] = DFO_PDA(L_aug, y_temp, d,nodes_num,gc,function_lambda,function_aalpha, features, labels,batch_size,iter_num);
  [xminuxbar_DDZO_PDA, sq_grad_DDZO_PDA, time_DDZO_PDA] = DDZO_PDA(L_aug, y_temp, d,nodes_num,gc,cf,function_lambda,function_aalpha, features, labels,batch_size,iter_num);
[xminuxbar_DFO_DGT, sq_grad_DFO_DGT, time_DFO_DGT] = DFO_DGT(W_aug, y_temp, d,nodes_num,gc,function_lambda,function_aalpha, features, labels,batch_size,iter_num);
 [xminuxbar_DDZO_DGT, sq_grad_DDZO_DGT, time_DDZO_DGT] = DDZO_DGT(W_aug, y_temp, d,nodes_num,gc,cf, function_lambda,function_aalpha, features, labels,batch_size,iter_num);
[Q,xminuxbar_xFILTER, sq_grad_xFILTER, time_xFILTER] = xFILTER(D, y_temp, edge_index,iter_num,big_L,  A, d,nodes_num,gc,function_lambda,function_aalpha, features, labels,batch_size);
[xminuxbar_GPDA, sq_grad_GPDA, time_GPDA] = GPDA(y_temp, edge_index,iter_num,big_L,WW,min_eig_L_hat, A,B,D,Adj,degree,d,nodes_num,gc,function_lambda,function_aalpha, features, labels,batch_size);
 [xminuxbar_Prox_GPDA, sq_grad_Prox_GPDA, time_Prox_GPDA] = Prox_PDA(y_temp, edge_index,iter_num,big_L,WW,min_eig_L_hat, A,B,D,Adj,degree,d,nodes_num,gc,function_lambda,function_aalpha, features, labels,batch_size);

sq_grad_DGD_FIXED_a = sq_grad_DGD_FIXED_a+sq_grad_DGD_FIXED/avera;
xminuxbar_DGD_FIXED_a = xminuxbar_DGD_FIXED_a+xminuxbar_DGD_FIXED/avera;
time_DGD_FIXED_a = time_DGD_FIXED_a+time_DGD_FIXED/avera;

sq_grad_DGD_DIMINISHING_a = sq_grad_DGD_DIMINISHING_a+sq_grad_DGD_DIMINISHING/avera;
xminuxbar_DGD_DIMINISHING_a =xminuxbar_DGD_DIMINISHING_a+ xminuxbar_DGD_DIMINISHING/avera;
time_DGD_DIMINISHING_a =time_DGD_DIMINISHING_a+ time_DGD_DIMINISHING/avera;

sq_grad_DFO_PDA_a =sq_grad_DFO_PDA_a+ sq_grad_DFO_PDA/avera;
xminuxbar_DFO_PDA_a =xminuxbar_DFO_PDA_a+ xminuxbar_DFO_PDA/avera;
time_DFO_PDA_a =time_DFO_PDA_a+ time_DFO_PDA/avera;

sq_grad_DDZO_PDA_a = sq_grad_DDZO_PDA_a+sq_grad_DDZO_PDA/avera;
xminuxbar_DDZO_PDA_a =xminuxbar_DDZO_PDA_a+ xminuxbar_DDZO_PDA/avera;
time_DDZO_PDA_a =time_DDZO_PDA_a+ time_DDZO_PDA/avera;

sq_grad_DFO_DGT_a = sq_grad_DFO_DGT_a+sq_grad_DFO_DGT/avera;
xminuxbar_DFO_DGT_a = xminuxbar_DFO_DGT_a+xminuxbar_DFO_DGT/avera;
time_DFO_DGT_a = time_DFO_DGT_a+time_DFO_DGT/avera;

sq_grad_DDZO_DGT_a = sq_grad_DDZO_DGT_a+sq_grad_DDZO_DGT/avera;
xminuxbar_DDZO_DGT_a = xminuxbar_DDZO_DGT_a+xminuxbar_DDZO_DGT/avera;
time_DDZO_DGT_a = time_DDZO_DGT_a+time_DDZO_DGT/avera;

sq_grad_xFILTER_a = sq_grad_xFILTER_a+sq_grad_xFILTER/avera;
xminuxbar_xFILTER_a = xminuxbar_xFILTER_a+xminuxbar_xFILTER/avera;
time_xFILTER_a = time_xFILTER_a+time_xFILTER/avera;

sq_grad_GPDA_a = sq_grad_GPDA_a+sq_grad_GPDA/avera;
xminuxbar_GPDA_a = xminuxbar_GPDA_a+xminuxbar_GPDA/avera;
time_GPDA_a = time_GPDA_a+time_GPDA/avera;

sq_grad_Prox_GPDA_a = sq_grad_Prox_GPDA_a+sq_grad_Prox_GPDA/avera;
xminuxbar_Prox_GPDA_a = xminuxbar_Prox_GPDA_a+xminuxbar_Prox_GPDA/avera;
time_Prox_GPDA_a = time_Prox_GPDA_a+time_Prox_GPDA/avera;

end


n=nodes_num;
ef=n^2;

T=iter_num;
tx=T*sum(Q(1,:));
xfilt_a=zeros(tx,1);
for i=1:T-1
    tq=sum(Q(1,1:i));
    for j=1:Q(1,i+1)      
       xfilt_a(tq+j)=sq_grad_xFILTER_a(i)/ef+xminuxbar_xFILTER_a(i)/n;
    end
end

figure;
semilogy(sq_grad_DFO_PDA_a/ef+xminuxbar_DFO_PDA_a/n, '-', 'LineWidth', 2); hold on;
semilogy(sq_grad_DDZO_PDA_a/ef+xminuxbar_DDZO_PDA_a/n, '--', 'LineWidth', 2);
semilogy(sq_grad_DGD_DIMINISHING_a/ef+xminuxbar_DGD_DIMINISHING_a/n, ':', 'LineWidth', 2);
semilogy(sq_grad_DFO_DGT_a/ef+xminuxbar_DFO_DGT_a/n, '-', 'LineWidth', 2);
semilogy(sq_grad_DDZO_DGT_a/ef+xminuxbar_DDZO_DGT_a/n, 'r--', 'LineWidth', 2);
semilogy(xfilt_a(1:T), '-.', 'Color','[0, 0.5, 0]', 'LineWidth', 2);
semilogy(sq_grad_Prox_GPDA_a/ef+xminuxbar_Prox_GPDA_a/n, '--', 'LineWidth', 2);
semilogy(sq_grad_GPDA_a/ef+xminuxbar_GPDA_a/n, ':', 'LineWidth', 2);

ylim([1e-35, 1e0]);
set(gca,'FontSize', 10);
xticks(0:100:iter_num);
xlabel('Number of communication rounds','Interpreter', 'latex', ...
        'FontSize', 15, 'FontWeight','bold');
ylabel('$\|\nabla f(\bar{x}_k)\|^ {2}+\frac{1}{n}\sum_{i=1}^{n}\|x_{i,k}-\bar{x}_k\|^ {2}$', 'Interpreter','latex', ...
        'FontSize', 15, 'FontWeight','bold');
legend({'Algorithm 1', ...
        'Algorithm 2', ...
        'DGD', ...
        'DFO-DGT', ...
        'DDZO-DGT', ...
       'xFILTER', ...
       'Prox-GPDA', ...
       'D-GPDA';}, ...
       'Interpreter', 'latex', 'FontSize', 10, 'FontWeight','bold');
