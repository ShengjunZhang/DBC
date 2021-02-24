% Chebyshev Update
% Solve d = R * x
% Input: R, d, x0
% Output: x

function [out, Q] = Chebyshev(R, d, x0, M, U)
    epsR = min(eig(R)) / max(eig(R));
    epsUR = min(eig(U*R)) / max(eig(U*R));
    epsU = min(eig(U)) / max(eig(U));

    Q = -1/4*log10(epsUR^2 * epsU^2/ (16+128*M*max(max(eig(R*U)), 1))) * sqrt(1/epsR);

    Q=floor(Q);
    t = 2 / (min(eig(R)) + max(eig(R)));
    alpha = 2;
    rho = (1 - epsR) / (1 + epsR);

    u(:, 1) = x0;
    u(:, 2) = (eye(size(R)) - t * R) * u(:, 1) + t * d;
    for i = 3: Q+3
        alpha = 4 / (4 - rho^2 * alpha);
        u(:, i) = alpha * (eye(size(R)) - t * R) * u(:, i-1) + (1- alpha) * u(:, i-2) + t * alpha * d;
    end
    out = u(:, end);
end
