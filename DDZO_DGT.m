function [xminuxbar, sq_grad,time] = DDZO_DGT(W_aug, x, n,N,gc,cf, lambda,aalpha, features, labels,bs, iter_num)

    fprintf('Starting DDZO_DGT\n');

    sq_grad = zeros(iter_num, 1);
    xminuxbar = zeros(iter_num, 1);
	time = zeros(iter_num, 1);

    gs = zeros(n*N, 1);
    ss = zeros(n*N, 1);
    

    eta1 = 1.3;
    u1 = 0.1;
    
    upd = textprogressbar(iter_num);
    
    for t = 2:iter_num
        tic;
        upd(t);
        eta = eta1;
        u = u1 / t^(3/4);
        
        prev_gs = gs;
        
        
        gradient = zeros(N*n,1);
        
        
        for ii = 1 : N 
            std_basis = eye(n);
            grad_temp = zeros(n, 1);
            for zz = 1:n
                for jj=(ii-1)*bs+1:ii*bs 
                    grad_temp =  grad_temp + (cf(x((ii-1)*n+1:ii*n,t-1)+u*std_basis(:, zz),lambda,aalpha, features(:,jj), labels(jj),bs, N) - cf(x((ii-1)*n+1:ii*n,t-1)-u*std_basis(:, zz),lambda,aalpha, features(:,jj), labels(jj),bs, N)).*std_basis(:, zz)/u/2;
                end 
            end
            
            gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + grad_temp;
        end
        
        gs = gradient;
        
        ss = W_aug* ( ss + gs - prev_gs ) ;
        
        x(:,t) = W_aug*( x(:, t -1) - eta*ss );
        
          
        xs = reshape(x(:,t), [n, N]);
        x_avg = sum(xs, 2) / N;
        
        for k = 1:N
            xminuxbar(t) = xminuxbar(t)+(norm(xs(:, k)-x_avg))^2;
%             fminufstar(t) = fminufstar(t)+loss_func(x_avg, y{k}, a_Re{k}, a_Im{k});
        end
        
        temp_grad = zeros(N*n,1);
        for ii = 1 : N
            for jj=(ii-1)*bs+1:ii*bs 
               temp_grad((ii-1)*n+1:ii*n) = temp_grad((ii-1)*n+1:ii*n) + gc(x_avg,lambda,aalpha, features(:,jj), labels(jj),bs, N); % This is compute the gradient in each node, batch_size works here.
            end
        end
        g = reshape(temp_grad, [n, N]);
        sq_grad(t) = sum(sum(g, 2).^2);
                    t_temp = toc;
    time(t) = time(t - 1) + t_temp;
    end
    
end