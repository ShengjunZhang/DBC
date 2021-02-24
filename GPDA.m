function [xminuxbar, sq_grad, time] = GPDA(x_temp, edge_index,iter_num,big_L,WW,min_eig_L_hat, A,B,D,Adj,degree,n,N,gc,lambda,aalpha, features, labels,bs)
%Opt_GPDA, full_grad, 
% % This is the original code from xFilter.

fprintf('GPDA starting... \n');
sq_grad = zeros(iter_num, 1); % T is the iteration number, we choose T = 1000;
xminuxbar = zeros(iter_num, 1);
time = zeros(iter_num, 1);
xs = x_temp(:,1);
xs_ = reshape(xs, [n, N]);
xs_temp = sum(xs_, 2);
temp_grad = zeros(N*n,1);
for ii = 1 : N
    for jj=(ii-1)*bs+1:ii*bs 
       temp_grad((ii-1)*n+1:ii*n) = temp_grad((ii-1)*n+1:ii*n) + gc(xs_temp,lambda,aalpha, features(:,jj), labels(jj),bs, N); % This is compute the gradient in each node, batch_size works here.
    end
end
% grad_temp = reshape(temp_grad, n, N); 
% 
% Opt_GPDA = zeros(iter_num-1,1);
x = x_temp;
mu = zeros((edge_index-1)*n,iter_num);
upd = textprogressbar(iter_num);

 beta = 80 * big_L * max(max(eig(WW)), 1) / (min(min_eig_L_hat, 1)*N);

    
    s_temp = zeros(size(A, 1)/n, 1);
    k=1;
    for ii = 1:size(Adj,1)
        for jj = ii+1:size(Adj,1)
            if Adj(ii, jj)==1
                s_temp(k) =  1/sqrt(degree(ii)*degree(jj));
                k=k+1;
            end
        end
    end
    sigma   = diag(beta * s_temp);
    sigma   = kron(sigma, eye(n));
    f1 = inv( A'*sigma *A + B'*sigma *B+ beta * eye(size(D)));
    f2 = B'* sigma * B + beta*eye(size(D));
     f1=sparse(f1);
     f2=sparse(f2);
     A=sparse(A);
     sigma=sparse(sigma);

for iter  = 2 : iter_num
    tic;
    upd(iter);
    

    
    % calculate the gradient
    gradient = zeros(N*n,1);
    gradient_matrix = zeros(n,N);
    for ii = 1 : N
        for jj=(ii-1)*bs+1:ii*bs
            gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x((ii-1)*n+1:ii*n,iter-1),lambda,aalpha, features(:,jj), labels(jj),bs, N);
        end
        gradient_matrix(:,ii) = gradient((ii-1)*n+1:ii*n);
    end
    
    % update x and mu
    x(:,iter) = f1*(f2* x(:,iter-1) - gradient - A.'*mu(:,iter-1));
    mu(:,iter) = mu(:,iter-1) + sigma*  A *x(:,iter);
    
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
    
    
%     % calculate opt
%     full_grad = sum(gradient_matrix,2);
%     Opt_GPDA(iter-1,1) = norm(full_grad)^2  + norm(A*x(:,iter))^2*big_L/N^2;
    t_temp = toc;
    time(iter) = time(iter - 1) + t_temp;
end