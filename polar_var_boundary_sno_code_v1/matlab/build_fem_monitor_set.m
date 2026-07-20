function build_fem_monitor_set(manifest_path, output_path)
%BUILD_FEM_MONITOR_SET Generate the fixed 100-case FEM validation set.

    S = load(manifest_path);
    cfg = struct();
    cfg.geom_base = scalar(S.geom_base);
    cfg.geom_amp = scalar(S.geom_amp);
    cfg.geom_tanh_scale = scalar(S.geom_tanh_scale);
    cfg.outer_scale = scalar(S.outer_scale);
    cfg.pcg_tol = scalar(S.pcg_tol);
    cfg.pcg_maxiter = round(scalar(S.pcg_maxiter));

    num_cases = size(S.geometry_w1, 1);
    theta_size = round(scalar(S.theta_size));
    radial_size = round(scalar(S.radial_size));
    eval_theta_size = round(scalar(S.eval_theta_size));
    eval_radial_size = round(scalar(S.eval_radial_size));
    mesh_levels = double(S.mesh_levels);
    convergence_tol = scalar(S.convergence_tol);

    pod_theta = linspace(0.0, 2.0*pi, theta_size + 1);
    pod_theta(end) = [];
    pod_eta = linspace(-1.0, 1.0, radial_size);
    [PodTheta, PodEta] = meshgrid(pod_theta, pod_eta);
    pod_coords = [rowmajor(PodTheta)/pi - 1.0, rowmajor(PodEta)];

    theta_edges = linspace(0.0, 2.0*pi, eval_theta_size + 1);
    eta_edges = linspace(-1.0, 1.0, eval_radial_size + 1);
    eval_theta = 0.5 * (theta_edges(1:end-1) + theta_edges(2:end));
    eval_eta = 0.5 * (eta_edges(1:end-1) + eta_edges(2:end));
    [EvalTheta, EvalEta] = meshgrid(eval_theta, eval_eta);
    eval_coords = [rowmajor(EvalTheta)/pi - 1.0, rowmajor(EvalEta)];

    boundary_coords = [pod_theta(:)/pi - 1.0, -ones(theta_size, 1)];
    n_pod = size(pod_coords, 1);
    n_eval = size(eval_coords, 1);
    p_pod = zeros(num_cases, n_pod);
    p_eval = zeros(num_cases, n_eval);
    area_weights = zeros(num_cases, n_eval);
    boundary_a = zeros(num_cases, theta_size);
    boundary_h = zeros(num_cases, theta_size);
    boundary_load = zeros(num_cases, theta_size);
    boundary_unit_flux = zeros(num_cases, theta_size);
    convergence_error = nan(num_cases, 1);
    mesh_level_used = zeros(num_cases, 1);
    pcg_relres = nan(num_cases, 1);
    pcg_iterations = nan(num_cases, 1);

    % Cross-language geometry check before the expensive solve.
    max_check_error = 0.0;
    check_theta = double(S.check_theta(:).');
    for icase = 1:num_cases
        geom = extract_geometry(S, icase);
        [a_check, adot_check] = eval_geometry_bnn(check_theta, geom, cfg);
        max_check_error = max(max_check_error, max(abs(a_check - S.check_a(icase, :))));
        max_check_error = max(max_check_error, max(abs(adot_check - S.check_a_theta(icase, :))));
    end
    if max_check_error > 2.0e-6
        error('Python/MATLAB geometry mismatch: %.3e', max_check_error);
    end
    fprintf('Geometry cross-check max error: %.3e\n', max_check_error);

    for icase = 1:num_cases
        geom = extract_geometry(S, icase);
        k = double(S.k_values(icase));
        fprintf('[FEM %03d/%03d] k=%.6f\n', icase, num_cases, k);

        [a_bnd, adot_bnd] = eval_geometry_bnn(pod_theta, geom, cfg);
        h_bnd = adot_bnd ./ a_bnd;
        g_bnd = cos(pod_theta) + h_bnd .* sin(pod_theta);
        boundary_a(icase, :) = a_bnd;
        boundary_h(icase, :) = h_bnd;
        boundary_load(icase, :) = g_bnd;
        boundary_unit_flux(icase, :) = g_bnd ./ sqrt(1.0 + h_bnd.^2);

        [a_eval, ~] = eval_geometry_bnn(EvalTheta, geom, cfg);
        radius_eval = a_eval .* (3.0 + 2.0*EvalEta);
        weights = 2.0 .* a_eval .* radius_eval;
        area_weights(icase, :) = rowmajor(weights).';

        previous_eval = [];
        accepted = false;
        last_fem = [];
        last_eval = [];
        for ilevel = 1:size(mesh_levels, 1)
            Nr = mesh_levels(ilevel, 1);
            Nt = mesh_levels(ilevel, 2);
            fem = solve_varpolar_fem(geom, k, Nr, Nt, cfg);
            if fem.flag ~= 0 || fem.relres > cfg.pcg_tol
                error('PCG failed for case %d level %d: flag=%d relres=%.3e', ...
                    icase, ilevel, fem.flag, fem.relres);
            end
            current_eval = interpolate_reference(fem, EvalTheta, EvalEta);
            if ~isempty(previous_eval)
                numerator = sum(weights(:) .* (current_eval(:)-previous_eval(:)).^2);
                denominator = max(sum(weights(:) .* current_eval(:).^2), eps);
                convergence_error(icase) = sqrt(numerator / denominator);
                fprintf('  level %d: Nr=%d Nt=%d convergence=%.3e\n', ...
                    ilevel, Nr, Nt, convergence_error(icase));
                if convergence_error(icase) <= convergence_tol
                    accepted = true;
                end
            end
            last_fem = fem;
            last_eval = current_eval;
            previous_eval = current_eval;
            mesh_level_used(icase) = ilevel;
            if accepted
                break;
            end
        end
        if ~accepted
            error('Case %d failed mesh convergence tolerance %.3e.', ...
                icase, convergence_tol);
        end

        p_eval(icase, :) = rowmajor(last_eval).';
        p_pod_values = interpolate_reference(last_fem, PodTheta, PodEta);
        p_pod(icase, :) = rowmajor(p_pod_values).';
        pcg_relres(icase) = last_fem.relres;
        pcg_iterations(icase) = last_fem.iter;
    end

    out = struct();
    out.pod_coords = pod_coords;
    out.eval_coords = eval_coords;
    out.boundary_coords = boundary_coords;
    out.p_pod = p_pod;
    out.p_eval = p_eval;
    out.area_weights = area_weights;
    out.boundary_a = boundary_a;
    out.boundary_h = boundary_h;
    out.boundary_load = boundary_load;
    out.boundary_unit_flux = boundary_unit_flux;
    out.k_values = double(S.k_values(:));
    out.convergence_error = convergence_error;
    out.mesh_level_used = mesh_level_used;
    out.pcg_relres = pcg_relres;
    out.pcg_iterations = pcg_iterations;
    out.geometry_w1 = S.geometry_w1;
    out.geometry_b1 = S.geometry_b1;
    out.geometry_w2 = S.geometry_w2;
    out.monitor_seed = S.monitor_seed;
    out.geometry_crosscheck_max_abs = max_check_error;

    output_dir = fileparts(output_path);
    if ~isempty(output_dir) && ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    save(output_path, '-struct', 'out', '-v7');
    fprintf('Saved FEM monitor set: %s\n', output_path);
end


function value = scalar(array)
    value = double(array(1));
end


function values = rowmajor(array)
    values = reshape(array.', [], 1);
end


function geom = extract_geometry(S, index)
    geom = struct();
    geom.w1 = squeeze(S.geometry_w1(index, :, :));
    geom.b1 = squeeze(S.geometry_b1(index, :));
    geom.w2 = squeeze(S.geometry_w2(index, :));
end


function values = interpolate_reference(fem, target_theta, target_eta)
    theta_closed = [fem.theta, 2.0*pi];
    values_closed = [fem.u_grid, fem.u_grid(:, 1)];
    target_scale = 3.0 + 2.0*target_eta;
    values = interp2(theta_closed, fem.scale, values_closed, ...
        mod(target_theta, 2.0*pi), target_scale, 'linear');
    if any(~isfinite(values(:)))
        error('Non-finite FEM interpolation result.');
    end
end
