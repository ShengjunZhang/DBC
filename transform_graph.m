function [W, L] = transform_graph(Adj, node_num)
    
    Ad = sparse(Adj);
    m = sum(sum(Ad))/2;
    
    A = zeros(node_num,m); %% A is the incidence matrix.
    l = 0;

    for i = 1:(node_num-1)
        for j = i+1:node_num
            if Ad(i,j) > 0.5
                l = l + 1;
                A(i,l) =  1;
                A(j,l) = -1;
            end
        end
    end

    W = mh_matrix(A);
    alpha = 100;
    eta =  0.7;
    L = 1/(alpha*eta) * (eye(node_num) - W);
    

end