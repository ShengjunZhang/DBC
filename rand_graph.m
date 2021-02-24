function [Adj, degree] = rand_graph(N, r)

    dim = 3;

    points = randn(N, dim);
    points = points ./ repmat(sqrt(sum(points.^2, 2)), 1, dim);

    sq_dists = zeros(N, N);

    for k = 1:3
        tmp = repmat(points(:, k), 1, N);
        sq_dists = sq_dists + (tmp-tmp').^2;
    end

    sq_dists = sq_dists + 3*eye(N);
    Adj=sq_dists <= (2*sin(r/2))^2;
    Adj_sparse = sparse(Adj);
    degs = sum(Adj_sparse + eye(N), 1);

    W = Adj_sparse ./ max(repmat(degs, N, 1), repmat(degs', 1, N));
    W = W + eye(N) - diag(sum(W, 1));
    degree = Adj * ones(N, 1);
end