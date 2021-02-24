function [Q,  xminuxbar, sq_grad, time] = xFILTER(D, x, edge_index,iter_num,big_L, A, n,N,gc,lambda,aalpha, features, labels,bs)
%Opt, full_grad, 
fprintf('xFilter starting...\n');
time = zeros(iter_num, 1);
sq_grad = zeros(iter_num, 1); % T is the iteration number, we choose T = 1000;
xminuxbar = zeros(iter_num, 1);
Q = zeros(iter_num, 1);
% fminufstar = zeros(iter_num, 1);
xs = x(:,1);
xs_ = reshape(xs, [n, N]);
xs_temp = sum(xs_, 2);
temp_grad = zeros(N*n,1);
for ii = 1 : N
    for jj=(ii-1)*bs+1:ii*bs 
       temp_grad((ii-1)*n+1:ii*n) = temp_grad((ii-1)*n+1:ii*n) + gc(xs_temp,lambda,aalpha, features(:,jj), labels(jj),bs, N); % This is compute the gradient in each node, batch_size works here.
    end
end
% grad_temp = reshape(temp_grad, n, N); 


% Parameter Chosen
% % I
k=2;
Lnorm = (D)^(-1/2) * (A' * A) * (D)^(-1/2);
beta = 96 * k * D * big_L / sum(sum(D));
eig_Lnorm = eig(Lnorm);
for ii = 1 : length(eig_Lnorm)
    if (eig_Lnorm(ii)>=1e-10)
        min_eig_Lnorm = eig_Lnorm(ii);
        break;
    end
end
sigma = 48* 96 * big_L/ (k*sum(sum(D))*min_eig_Lnorm);

% % % % II
% beta = 96*big_L/N;
% Lnorm = A' * A  ;
% eig_Lnorm = eig(Lnorm);
% for ii = 1 : length(eig_Lnorm)
%     if (eig_Lnorm(ii)>=1e-10)
%         min_eig_Lnorm = eig_Lnorm(ii);
%         break;
%     end
% end
% sigma = 48* beta / min_eig_Lnorm;

% Iterative Update
mu = zeros((edge_index-1)*n,iter_num);
% Opt = zeros(iter_num-1,1);

% temp_grad = gc(x,lambda,aalpha, features, labels,bs, N);

upd = textprogressbar(iter_num);
for iter  = 2 : iter_num
    tic;
    upd(iter);
    % Calculate the gradient
    gradient = zeros(N*n,1);
    for ii = 1 : N
        for jj=(ii-1)*bs+1:ii*bs 
            gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x((ii-1)*n+1:ii*n,iter-1),lambda,aalpha, features(:,jj), labels(jj),bs, N); % This is compute the gradient in each node, batch_size works here.
        end
    end
%     gradient_matrix = reshape(gradient, n, N);
    
    % Update x, and mu
    R = beta^(-1) * (sigma * (A' * A)) + eye(N*n);
    d = x(:,iter-1) - beta^(-1) * gradient - beta^(-1) * A' * mu(:,iter-1);
    [x(:,iter), Q(:,iter)] = Chebyshev(R, d, x(:,iter-1), N, beta);
    mu(:,iter) = mu(:,iter-1) + sigma * (A*x(:,iter));
    
%     x_avg = sum(x_, 2)/n_agents;
%         
%     for k = 1: n_agents
%         xminuxbar(iter) = xminuxbar(iter)+(norm(x(:, k)-x_avg))^2;
%     end
%         tmp_grad = gc(x_avg,lambda,aalpha, features, labels,bs, N);
%         sq_grad(iter) = sum(tmp_grad.^2);
%     
    
    x_ = reshape(x(:, iter), [n, N]);
    
    x_avg = sum(x_, 2)/N;
    
    for k = 1:N
        xminuxbar(iter) = xminuxbar(iter)+(norm(x_(:, k)-x_avg))^2;
    end
    
    % Compute the sq_grad
    
    temp_grad = zeros(N*n,1);
    for ii = 1 : N
        for jj=(ii-1)*bs+1:ii*bs 
           temp_grad((ii-1)*n+1:ii*n) = temp_grad((ii-1)*n+1:ii*n) + gc(x_avg,lambda,aalpha, features(:,jj), labels(jj),bs, N); % This is compute the gradient in each node, batch_size works here.
        end
    end
    g = reshape(temp_grad, [n, N]);
    sq_grad(iter) = sum(sum(g, 2).^2);
    
%     % Calculate opt
%     full_grad = sum(gradient_matrix,2);
%     Opt(iter-1,1) =  norm(full_grad)^2  + norm(A*x(:,iter))^2*big_L/N^2;
    t_temp = toc;
    time(iter) = time(iter - 1) + t_temp;    
end
