function fem = solve_varpolar_fem(geom, k, Nr, Nt, cfg)
%SOLVE_VARPOLAR_FEM Boundary-fitted P1 FEM for one variable annulus.
%
% Solves -Delta P + k^2 P = 0 with P=0 on r=5a(theta). The inner
% boundary load supplied to the weak form is the unit computational-domain
% outward flux
%
%   q_Omega = g_target / sqrt(1 + (a_theta/a)^2),
%   g_target = cos(theta) + (a_theta/a) sin(theta).

    theta = linspace(0.0, 2.0*pi, Nt + 1);
    theta(end) = [];
    scale = linspace(1.0, double(cfg.outer_scale), Nr);
    [Theta, Scale] = meshgrid(theta, scale);
    a_theta_nodes = eval_geometry_bnn(theta, geom, cfg);
    Radius = Scale .* repmat(a_theta_nodes, Nr, 1);
    X = Radius .* cos(Theta);
    Y = Radius .* sin(Theta);
    nodes = [X(:), Y(:)];

    id = @(ir, it) (it - 1) * Nr + ir;
    num_elems = 2 * (Nr - 1) * Nt;
    elems = zeros(num_elems, 3);
    cursor = 0;
    for it = 1:Nt
        itp = mod(it, Nt) + 1;
        for ir = 1:(Nr - 1)
            n00 = id(ir, it);
            n10 = id(ir + 1, it);
            n01 = id(ir, itp);
            n11 = id(ir + 1, itp);
            cursor = cursor + 1;
            elems(cursor, :) = [n00, n10, n11];
            cursor = cursor + 1;
            elems(cursor, :) = [n00, n11, n01];
        end
    end

    num_nodes = size(nodes, 1);
    I = zeros(9 * num_elems, 1);
    J = zeros(9 * num_elems, 1);
    V = zeros(9 * num_elems, 1);
    F = zeros(num_nodes, 1);
    ptr = 0;
    min_area = inf;

    for e = 1:num_elems
        vid = elems(e, :);
        xe = nodes(vid, 1);
        ye = nodes(vid, 2);
        determinant = det([xe(2)-xe(1), xe(3)-xe(1); ...
                           ye(2)-ye(1), ye(3)-ye(1)]);
        area = 0.5 * abs(determinant);
        min_area = min(min_area, area);
        if ~isfinite(area) || area <= 0.0
            error('Degenerate triangle detected in geometry mesh.');
        end
        b = [ye(2)-ye(3); ye(3)-ye(1); ye(1)-ye(2)];
        c = [xe(3)-xe(2); xe(1)-xe(3); xe(2)-xe(1)];
        Kloc = (b*b' + c*c') / (4.0 * area);
        Mloc = area / 12.0 * [2 1 1; 1 2 1; 1 1 2];
        Aloc = Kloc + double(k)^2 * Mloc;
        for ia = 1:3
            for ib = 1:3
                ptr = ptr + 1;
                I(ptr) = vid(ia);
                J(ptr) = vid(ib);
                V(ptr) = Aloc(ia, ib);
            end
        end
    end
    A = sparse(I, J, V, num_nodes, num_nodes);

    % Inner boundary Neumann contribution.
    for it = 1:Nt
        itp = mod(it, Nt) + 1;
        n1 = id(1, it);
        n2 = id(1, itp);
        edge_len = hypot(nodes(n2,1)-nodes(n1,1), nodes(n2,2)-nodes(n1,2));
        theta_next = theta(itp);
        dtheta = mod(theta_next - theta(it) + pi, 2*pi) - pi;
        theta_mid = mod(theta(it) + 0.5*dtheta, 2*pi);
        [a_mid, adot_mid] = eval_geometry_bnn(theta_mid, geom, cfg);
        h_mid = adot_mid / a_mid;
        g_target = cos(theta_mid) + h_mid * sin(theta_mid);
        q_omega = g_target / sqrt(1.0 + h_mid^2);
        F(n1) = F(n1) + q_omega * edge_len / 2.0;
        F(n2) = F(n2) + q_omega * edge_len / 2.0;
    end

    outer_nodes = (Nr:Nr:num_nodes).';
    all_nodes = (1:num_nodes).';
    free_nodes = setdiff(all_nodes, outer_nodes);
    Aff = double(A(free_nodes, free_nodes));
    Ff = double(F(free_nodes));

    opts = struct('type', 'ict', 'droptol', 1.0e-3, 'diagcomp', 1.0e-3);
    L = ichol(Aff, opts);
    [Uf, flag, relres, iter, resvec] = pcg(...
        Aff, Ff, double(cfg.pcg_tol), double(cfg.pcg_maxiter), ...
        L, L', zeros(size(Ff)));

    U = zeros(num_nodes, 1);
    U(free_nodes) = Uf;
    fem = struct();
    fem.x = X;
    fem.y = Y;
    fem.u_grid = reshape(U, Nr, Nt);
    fem.theta = theta;
    fem.scale = scale;
    fem.nodes = nodes;
    fem.elems = elems;
    fem.flag = flag;
    fem.relres = relres;
    fem.iter = iter;
    fem.resvec = resvec;
    fem.min_area = min_area;
end
