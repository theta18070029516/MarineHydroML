function fem = solve_annulus_star_fem(geom, k, Nr, Nt)

    %% ------------------------------------------------------------
    % Build boundary-fitted mesh
    %% ------------------------------------------------------------

    theta = linspace(0, 2*pi, Nt + 1);
    theta(end) = [];

    scale = linspace(1.0, geom.outer_scale, Nr);

    [Theta, Scale] = meshgrid(theta, scale);

    [a_theta, ~] = geom_radius_derivative(theta, geom);

    Agrid = repmat(a_theta(:).', Nr, 1);

    R = Scale .* Agrid;

    X = R .* cos(Theta);
    Y = R .* sin(Theta);

    nodes = [X(:), Y(:)];

    id = @(ir, it) (it - 1) * Nr + ir;

    elems = zeros(2 * (Nr - 1) * Nt, 3);
    ecount = 0;

    for it = 1:Nt
        itp = mod(it, Nt) + 1;

        for ir = 1:(Nr - 1)
            n00 = id(ir, it);
            n10 = id(ir + 1, it);
            n01 = id(ir, itp);
            n11 = id(ir + 1, itp);

            ecount = ecount + 1;
            elems(ecount, :) = [n00, n10, n11];

            ecount = ecount + 1;
            elems(ecount, :) = [n00, n11, n01];
        end
    end

    num_nodes = size(nodes, 1);
    num_elems = size(elems, 1);

    %% ------------------------------------------------------------
    % Assemble FEM matrix
    % Solve: -Delta u + k^2 u = 0
    %% ------------------------------------------------------------

    I = zeros(9 * num_elems, 1);
    J = zeros(9 * num_elems, 1);
    V = zeros(9 * num_elems, 1);
    ptr = 0;

    F = zeros(num_nodes, 1);

    for e = 1:num_elems

        vid = elems(e, :);

        xe = nodes(vid, 1);
        ye = nodes(vid, 2);

        x1 = xe(1); x2 = xe(2); x3 = xe(3);
        y1 = ye(1); y2 = ye(2); y3 = ye(3);

        area = 0.5 * abs(det([x2 - x1, x3 - x1; ...
                              y2 - y1, y3 - y1]));

        if area <= 0
            error('Degenerate triangle detected.');
        end

        b = [y2 - y3; y3 - y1; y1 - y2];
        c = [x3 - x2; x1 - x3; x2 - x1];

        Kloc = (b * b' + c * c') / (4.0 * area);
%         Kloc = area * Kloc;

        Mloc = area / 12.0 * [2 1 1; 1 2 1; 1 1 2];

        Aloc = Kloc + k^2 * Mloc;

        for a = 1:3
            for bidx = 1:3
                ptr = ptr + 1;
                I(ptr) = vid(a);
                J(ptr) = vid(bidx);
                V(ptr) = Aloc(a, bidx);
            end
        end
    end

    A = sparse(I, J, V, num_nodes, num_nodes);

    %% ------------------------------------------------------------
    % Inner Neumann boundary contribution
    %
    % Model boundary convention:
    %   g = (e_r - a'/a e_theta) · grad u
    %
    % FEM weak form uses unit outward normal of computational domain.
    % On inner boundary:
    %   q_FEM = n_domain · grad u = - g / sqrt(1+(a'/a)^2)
    %% ------------------------------------------------------------

    for it = 1:Nt

        itp = mod(it, Nt) + 1;

        n1 = id(1, it);
        n2 = id(1, itp);

        x1 = nodes(n1, 1);
        y1 = nodes(n1, 2);
        x2 = nodes(n2, 1);
        y2 = nodes(n2, 2);

        edge_len = sqrt((x2 - x1)^2 + (y2 - y1)^2);

        th1 = theta(it);
        th2 = theta(itp);

        dth = angle_diff(th2, th1);
        th_mid = th1 + 0.5 * dth;
        th_mid = mod(th_mid, 2*pi);

        [a_mid, adot_mid] = geom_radius_derivative(th_mid, geom);

        g_mid = cos(th_mid) + (adot_mid / a_mid) * sin(th_mid);

%         metric = sqrt(1.0 + (adot_mid / a_mid)^2);

%         q_fem = -g_mid / metric;
        q_fem = g_mid;

        F(n1) = F(n1) + q_fem * edge_len / 2.0;
        F(n2) = F(n2) + q_fem * edge_len / 2.0;

    end

    %% ------------------------------------------------------------
    % Outer Dirichlet boundary: u = 0
    %% ------------------------------------------------------------

    outer_nodes = zeros(Nt, 1);
    for it = 1:Nt
        outer_nodes(it) = id(Nr, it);
    end

    all_nodes = (1:num_nodes).';
    free_nodes = setdiff(all_nodes, outer_nodes);

    U = zeros(num_nodes, 1);

%     U(free_nodes) = A(free_nodes, free_nodes) \ F(free_nodes);
    
    Aff = A(free_nodes, free_nodes);
    Ff  = F(free_nodes);

    % Ensure double precision
    Aff = double(Aff);
    Ff  = double(Ff);

    % PCG parameters
    tol = 1.0e-10;
    maxit = 1000;

    % Initial guess
    x0 = zeros(size(Ff));

    % Incomplete Cholesky preconditioner
    opts = struct();
    opts.type = 'ict';
    opts.droptol = 1.0e-3;
    opts.diagcomp = 1.0e-3;

    L = ichol(Aff, opts);

    [Uf, flag, relres, iter, resvec] = pcg(Aff, Ff, tol, maxit, L, L', x0);

    if flag ~= 0
        warning('PCG did not fully converge: flag=%d, relres=%.3e, iter=%d', ...
            flag, relres, iter);
    else
        fprintf('PCG converged: relres=%.3e, iter=%d\n', relres, iter);
    end

    U = zeros(num_nodes, 1);
    U(free_nodes) = Uf;

    %% ------------------------------------------------------------
    % Reshape solution
    %% ------------------------------------------------------------

    Ugrid = reshape(U, Nr, Nt);

    fem.x = X;
    fem.y = Y;
    fem.u_grid = Ugrid;
    fem.theta = theta;
    fem.nodes = nodes;
    fem.elems = elems;

end


function [a, adot] = geom_radius_derivative(theta, geom)

    switch lower(geom.type)

        case 'circle'
            a = geom.r_inner + zeros(size(theta));
            adot = zeros(size(theta));

        case 'square'
            [a, adot] = square_radius_derivative(theta, geom.corner_radius);

        case 'polygon'
            [a, adot] = polygon_radius_derivative( ...
                theta, ...
                geom.n_sides, ...
                geom.corner_radius, ...
                geom.rotation);

        otherwise
            error('Unknown geometry type: %s', geom.type);
    end

end


function [a, adot] = square_radius_derivative(theta, corner_radius)

    h = corner_radius / sqrt(2.0);

    c = cos(theta);
    s = sin(theta);

    abs_c = abs(c);
    abs_s = abs(s);

    use_cos_branch = abs_c >= abs_s;

    a = h ./ max(abs_c, abs_s);

    adot = zeros(size(theta));

    mask_c = use_cos_branch;
    adot(mask_c) = ...
        h .* s(mask_c) .* sign(c(mask_c)) ./ ...
        (abs_c(mask_c).^2 + 1.0e-12);

    mask_s = ~use_cos_branch;
    adot(mask_s) = ...
        -h .* c(mask_s) .* sign(s(mask_s)) ./ ...
        (abs_s(mask_s).^2 + 1.0e-12);

    corner_mask = abs(abs_c - abs_s) < 1.0e-10;
    adot(corner_mask) = 0.0;

end


function [a, adot] = polygon_radius_derivative(theta, n_sides, corner_radius, rotation)

    theta = theta(:);

    n = n_sides;
    Rc = corner_radius;

    apothem = Rc * cos(pi / n);

    alpha_all = rotation + (2 * (0:n-1) + 1) * pi / n;

    delta_all = zeros(numel(theta), n);

    for j = 1:n
        delta_all(:, j) = angle_diff(theta, alpha_all(j));
    end

    [~, idx] = min(abs(delta_all), [], 2);

    alpha = alpha_all(idx).';

    delta = angle_diff(theta, alpha);

    cos_delta = cos(delta);

    a = apothem ./ cos_delta;

    adot = apothem .* sin(delta) ./ (cos_delta.^2 + 1.0e-12);

    vertex_mask = abs(abs(delta) - pi/n) < 1.0e-10;
    adot(vertex_mask) = 0.0;

    a = reshape(a, size(theta));
    adot = reshape(adot, size(theta));

end


function d = angle_diff(a, b)
    d = atan2(sin(a - b), cos(a - b));
end


function A = close_theta(A)
    A = [A, A(:, 1)];
end


function plot_field(X, Y, Z, cmin, cmax, ttl)

    h = surf(X, Y, Z);

    set(h, ...
        'EdgeColor', 'none', ...
        'LineStyle', 'none', ...
        'FaceColor', 'interp');

    view(2);
    axis equal tight;
    box on;
    grid off;

    set(gca, ...
        'FontName', 'Arial', ...
        'FontSize', 10, ...
        'LineWidth', 0.85, ...
        'TickDir', 'out', ...
        'Layer', 'top', ...
        'XGrid', 'off', ...
        'YGrid', 'off', ...
        'ZGrid', 'off');

    caxis([cmin, cmax]);

    cb = colorbar;
    cb.FontSize = 9;
    cb.TickDirection = 'out';

    title(ttl, ...
        'Interpreter', 'latex', ...
        'FontSize', 12);

    xlabel('$x$', 'Interpreter', 'latex');
    ylabel('$y$', 'Interpreter', 'latex');

end