function W = mh_matrix(A)
% Computes the Metropolis-Hastings weights matrix
%
% WEIGHT_MATRIX = MH_MATRIX(A) gives a matrix of the Metropolis-Hastings 
% edge weights for a graphb described by the incidence matrix A (NxM). 
% N is the number of nodes, and M is the number of edges. 
% Each column of A has exactly one
% +1 and one -1.
% The M.-H. weight on an edge is one over the maximum of the degrees of the
% adjacent nodes.
%
% For more details, see the references:
% "Fast linear iterations for distributed averaging" by L. Xiao and S. Boyd
% "Fastest mixing Markov chain on a graph" by S. Boyd, P. Diaconis, 
%  and L. Xiao
% "Convex Optimization of Graph Laplacian Eigenvalues" by S. Boyd
%
% Daniel Zhang 12/02/2019

% degrees of the nodes
[n, ~] = size(A);
W = zeros(n, n);
Lunw = A*A';          % unweighted Laplacian matrix
Lunw = abs(Lunw);

for i = 1:(n-1)
    for j = i+1:n
        if Lunw(i, j) == 1
            W(i,j) = 1/(1 + max(Lunw(i,i), Lunw(j,j)));
        end
    end
end

W = W + W';
sum_row_W = sum(W);
for i = 1:n
    W(i,i) = 1 - sum_row_W(i);
end
