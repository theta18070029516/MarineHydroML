clear; clc; close all;

%% ============================================================
% Case configuration
%% ============================================================

cases = struct([]);

% % Circle
% cases(1).name = 'Circle';
% cases(1).file = 'circle_f0_flux_cos_theta_transformer_fe_k_0p5_1p0_1p5_fem_refgrid.mat';
% cases(1).geom.type = 'circle';
% cases(1).geom.r_inner = 0.2;
% cases(1).geom.outer_scale = 5.0;

% Square
cases(1).name = 'Square';
cases(1).file = 'square_corner_radius_0p2_f0_problem_flux_transformer_fe_k_0p5_1p0_1p5_fem_refgrid.mat';
cases(1).geom.type = 'square';
cases(1).geom.corner_radius = 0.2;
cases(1).geom.outer_scale = 5.0;
% 
% % Pentagon
% cases(1).name = 'Pentagon';
% cases(1).file = 'pentagon_vertex_on_y_corner_radius_0p2_f0_problem_flux_transformer_fe_k_0p5_1p0_1p5_fem_refgrid.mat';
% cases(1).geom.type = 'polygon';
% cases(1).geom.n_sides = 5;
% cases(1).geom.corner_radius = 0.2;
% cases(1).geom.rotation = pi/10;      % one vertex on +y axis
% cases(1).geom.outer_scale = 5.0;

%% ============================================================
% PCG settings
%% ============================================================

pcg_opt.tol = 1.0e-8;
pcg_opt.maxit = 2000;

pcg_opt.ichol.type = 'ict';
pcg_opt.ichol.droptol = 1.0e-3;
pcg_opt.ichol.diagcomp = 1.0e-3;

%% ============================================================
% Plot style
%% ============================================================

style.fontName = 'Arial';
style.axisFontSize = 9.5;
style.titleFontSize = 11;
style.sgTitleFontSize = 13;
style.cbFontSize = 8.5;
style.lineWidth = 0.85;

%% ============================================================
% Result table storage
%% ============================================================

result_rows = {};

%% ============================================================
% Main loop over shapes
%% ============================================================

for icase_shape = 1:numel(cases)

    case_name = cases(icase_shape).name;
    case_file = cases(icase_shape).file;
    geom = cases(icase_shape).geom;

    fprintf('\n============================================================\n');
    fprintf('Case: %s\n', case_name);
    fprintf('File: %s\n', case_file);
    fprintf('============================================================\n');

    S = load(case_file);

    if ~isfield(S, 'u_pred_ref_grid')
        error('File %s does not contain u_pred_ref_grid.', case_file);
    end

    if ~isfield(S, 'x_phys_ref_grid') || ~isfield(S, 'y_phys_ref_grid')
        error('File %s must contain x_phys_ref_grid and y_phys_ref_grid.', case_file);
    end

    k_values = double(S.k_values(:));
    num_k = numel(k_values);

    U_pred_ref_all = double(S.u_pred_ref_grid);   % [num_k, Nr_ref, Nt_ref]
    Xref_mat = double(S.x_phys_ref_grid);
    Yref_mat = double(S.y_phys_ref_grid);

    Nr_ref = size(Xref_mat, 1);
    Nt_ref = size(Xref_mat, 2);

    if isfield(S, 'Nr_ref')
        Nr_ref_file = double(S.Nr_ref);
        Nt_ref_file = double(S.Nt_ref);

        if Nr_ref ~= Nr_ref_file || Nt_ref ~= Nt_ref_file
            warning('Nr_ref/Nt_ref fields do not match coordinate grid size.');
        end
    end

    fprintf('Ref grid: Nr_ref=%d, Nt_ref=%d, N=%d\n', ...
        Nr_ref, Nt_ref, Nr_ref * Nt_ref);

    %% ------------------------------------------------------------
    % Loop over k
    %% ------------------------------------------------------------

    for ik = 1:num_k

        k = k_values(ik);

        fprintf('\n[%s] k = %.4f\n', case_name, k);

        %% --------------------------------------------------------
        % 1. Assemble FEM system once
        %% --------------------------------------------------------

        t_assemble = tic;
        fem = assemble_annulus_star_fem_system(geom, k, Nr_ref, Nt_ref);
        time_assemble = toc(t_assemble);

        % Verify the FEM grid matches the exported ref grid.
        grid_mismatch = max(abs(fem.x(:) - Xref_mat(:))) + ...
                        max(abs(fem.y(:) - Yref_mat(:)));

        fprintf('Grid mismatch between FEM and MAT ref grid: %.3e\n', grid_mismatch);

        if grid_mismatch > 1.0e-8
            warning(['FEM grid and exported ref grid are not exactly identical. ', ...
                     'Check geometry parameters, rotation, and outer_scale.']);
        end

        Aff = fem.A(fem.free_nodes, fem.free_nodes);
        Ff  = fem.F(fem.free_nodes);

        Aff = double(Aff);
        Ff  = double(Ff);

        %% --------------------------------------------------------
        % 2. Build one common incomplete Cholesky preconditioner
        %% --------------------------------------------------------

        t_ichol = tic;

        try
            L = ichol(Aff, pcg_opt.ichol);
        catch ME
            warning('ichol failed with default settings. Retrying with stronger diagcomp.');
            pcg_opt_retry = pcg_opt;
            pcg_opt_retry.ichol.droptol = 1.0e-2;
            pcg_opt_retry.ichol.diagcomp = 1.0e-1;
            L = ichol(Aff, pcg_opt_retry.ichol);
        end

        time_ichol = toc(t_ichol);

        %% --------------------------------------------------------
        % 3. Prepare initial guesses
        %% --------------------------------------------------------

        U_pred_grid = squeeze(U_pred_ref_all(ik, :, :));   % [Nr_ref, Nt_ref]

        if size(U_pred_grid, 1) ~= Nr_ref || size(U_pred_grid, 2) ~= Nt_ref
            error('u_pred_ref_grid size does not match FEM grid.');
        end

        U_pred_all_vec = U_pred_grid(:);
        x0_pred = U_pred_all_vec(fem.free_nodes);

        x0_zero = zeros(size(Ff));

        %% --------------------------------------------------------
        % 4. PCG with zero initial guess
        %% --------------------------------------------------------

        t_zero = tic;
        [Uf_zero, flag_zero, relres_zero, iter_zero, resvec_zero] = pcg( ...
            Aff, Ff, ...
            pcg_opt.tol, pcg_opt.maxit, ...
            L, L', ...  % L, L'
            x0_zero);
        time_zero = toc(t_zero);

        %% --------------------------------------------------------
        % 5. PCG with model prediction initial guess
        %% --------------------------------------------------------

        t_pred = tic;
        [Uf_predinit, flag_pred, relres_pred, iter_pred, resvec_pred] = pcg( ...
            Aff, Ff, ...
            pcg_opt.tol, pcg_opt.maxit, ...
            L, L', ...  % L, L'
            x0_pred);
        time_pred = toc(t_pred);

        %% --------------------------------------------------------
        % 6. Reconstruct full-grid FEM solutions
        %% --------------------------------------------------------

        U_zero_vec = zeros(Nr_ref * Nt_ref, 1);
        U_zero_vec(fem.free_nodes) = Uf_zero;
        U_zero_grid = reshape(U_zero_vec, Nr_ref, Nt_ref);

        U_predinit_vec = zeros(Nr_ref * Nt_ref, 1);
        U_predinit_vec(fem.free_nodes) = Uf_predinit;
        U_predinit_grid = reshape(U_predinit_vec, Nr_ref, Nt_ref);

        %% --------------------------------------------------------
        % 7. Error metrics
        %% --------------------------------------------------------

        err_model_vs_fem = norm(U_pred_grid(:) - U_zero_grid(:), 2) / ...
                           (norm(U_zero_grid(:), 2) + eps);

        err_fem_predinit_vs_zero = norm(U_predinit_grid(:) - U_zero_grid(:), 2) / ...
                                   (norm(U_zero_grid(:), 2) + eps);

        init_res_zero = norm(Ff - Aff * x0_zero, 2) / (norm(Ff, 2) + eps);
        init_res_pred = norm(Ff - Aff * x0_pred, 2) / (norm(Ff, 2) + eps);

        speedup_iter = iter_zero / max(iter_pred, 1);
        speedup_time = time_zero / max(time_pred, eps);

        fprintf('Assembly time: %.3f s, ichol time: %.3f s\n', ...
            time_assemble, time_ichol);

        fprintf('Zero init : flag=%d, iter=%d, relres=%.3e, pcg_time=%.3f s, init_res=%.3e\n', ...
            flag_zero, iter_zero, relres_zero, time_zero, init_res_zero);

        fprintf('Pred init : flag=%d, iter=%d, relres=%.3e, pcg_time=%.3f s, init_res=%.3e\n', ...
            flag_pred, iter_pred, relres_pred, time_pred, init_res_pred);

        fprintf('Model vs FEM relerr = %.3e\n', err_model_vs_fem);
        fprintf('FEM(pred-init) vs FEM(zero-init) relerr = %.3e\n', err_fem_predinit_vs_zero);
        fprintf('Iter speedup = %.2f, time speedup = %.2f\n', ...
            speedup_iter, speedup_time);

        result_rows(end+1, :) = { ...
            case_name, k, ...
            Nr_ref, Nt_ref, ...
            iter_zero, time_zero, flag_zero, relres_zero, init_res_zero, ...
            iter_pred, time_pred, flag_pred, relres_pred, init_res_pred, ...
            speedup_iter, speedup_time, ...
            err_model_vs_fem, err_fem_predinit_vs_zero, ...
            time_assemble, time_ichol};

        %% --------------------------------------------------------
        % 8. Plot 2 x 3 comparison figure
        %% --------------------------------------------------------

        plot_warmstart_comparison_refgrid( ...
            Xref_mat, Yref_mat, ...
            U_pred_grid, U_zero_grid, U_predinit_grid, ...
            case_name, k, ...
            iter_zero, time_zero, relres_zero, ...
            iter_pred, time_pred, relres_pred, ...
            err_model_vs_fem, err_fem_predinit_vs_zero, ...
            style);

        %% --------------------------------------------------------
        % Optional: residual curve
        %% --------------------------------------------------------
        figure('Color', 'w', ...
               'Units', 'centimeters', ...
               'Position', [4, 4, 12, 8]);

        semilogy(resvec_zero./ resvec_zero(1) , 'LineWidth', 1.8); hold on; %./ resvec_zero(1)
        semilogy(resvec_pred./ resvec_zero(1) , 'LineWidth', 1.8);  %./ resvec_pred(1)
        hold off;

        grid on;
        box on;

        set(gca, ...
            'FontName', style.fontName, ...
            'FontSize', 10, ...
            'LineWidth', 0.9, ...
            'TickDir', 'out');

        xlabel('PCG iteration', 'Interpreter', 'latex');
        ylabel('Relative residual', 'Interpreter', 'latex');
        legend({'zero init', 'prediction init'}, ...
            'Interpreter', 'latex', ...
            'Location', 'southwest', ...
            'Box', 'off');

        title(sprintf('%s, $k=%.2f$', case_name, k), ...
            'Interpreter', 'latex', ...
            'FontSize', 12);

    end

end

%% ============================================================
% Summary table
%% ============================================================

ResultTable = cell2table(result_rows, ...
    'VariableNames', { ...
    'shape', 'k', ...
    'Nr_ref', 'Nt_ref', ...
    'iter_zero', 'time_zero', 'flag_zero', 'relres_zero', 'init_res_zero', ...
    'iter_pred', 'time_pred', 'flag_pred', 'relres_pred', 'init_res_pred', ...
    'speedup_iter', 'speedup_time', ...
    'relerr_model_vs_fem', 'relerr_fem_predinit_vs_zero', ...
    'time_assemble', 'time_ichol'});

disp(ResultTable);


function plot_warmstart_comparison_refgrid( ...
    X, Y, ...
    U_pred, U_zero, U_predinit, ...
    case_name, k, ...
    iter_zero, time_zero, relres_zero, ...
    iter_pred, time_pred, relres_pred, ...
    err_model_vs_fem, err_fem_predinit_vs_zero, ...
    style)

    Err_pred_zero = abs(U_pred - U_zero);
    Err_predinit_zero = abs(U_predinit - U_zero);

    Xc = close_theta(X);
    Yc = close_theta(Y);

    U_pred_c = close_theta(U_pred);
    U_zero_c = close_theta(U_zero);
    U_predinit_c = close_theta(U_predinit);

    Err_pred_zero_c = close_theta(Err_pred_zero);
    Err_predinit_zero_c = close_theta(Err_predinit_zero);

    % Shared solution colorbar for all solution fields
    u_min = min([U_pred_c(:); U_zero_c(:); U_predinit_c(:)]);
    u_max = max([U_pred_c(:); U_zero_c(:); U_predinit_c(:)]);

    % Shared error colorbar for both error fields
    err_max = max([Err_pred_zero_c(:); Err_predinit_zero_c(:)]);
    if err_max <= 0
        err_max = eps;
    end

    fig = figure('Color', 'w', ...
                 'Units', 'centimeters', ...
                 'Position', [2, 2, 32, 17]);

    tiledlayout(2, 3, ...
        'TileSpacing', 'compact', ...
        'Padding', 'loose');

    colormap(turbo);

    % ------------------------------------------------------------
    % Row 1
    % Prediction, FEM from zero init, error
    % ------------------------------------------------------------

    nexttile(1);
    plot_field_sci(Xc, Yc, U_pred_c, u_min, u_max, ...
        '$u_{\mathrm{pred}}$', style);

    nexttile(2);
    plot_field_sci(Xc, Yc, U_zero_c, u_min, u_max, ...
        '$u_{\mathrm{FEM}}^{0}$', style);

    nexttile(3);
    plot_field_sci(Xc, Yc, Err_pred_zero_c, 0, err_max, ...
        '$|u_{\mathrm{pred}}-u_{\mathrm{FEM}}^{0}|$', style);

    % ------------------------------------------------------------
    % Row 2
    % FEM from prediction init, FEM from zero init, error
    % ------------------------------------------------------------

    nexttile(4);
    plot_field_sci(Xc, Yc, U_predinit_c, u_min, u_max, ...
        '$u_{\mathrm{FEM}}^{\mathrm{pred\ init}}$', style);

    nexttile(5);
    plot_field_sci(Xc, Yc, U_zero_c, u_min, u_max, ...
        '$u_{\mathrm{FEM}}^{0}$', style);

    nexttile(6);
    plot_field_sci(Xc, Yc, Err_predinit_zero_c, 0, err_max, ...
        '$|u_{\mathrm{FEM}}^{\mathrm{pred\ init}}-u_{\mathrm{FEM}}^{0}|$', style);

    sgtitle(sprintf(['%s, $k=%.2f$  |  ', ...
                     '$E_{pred}=%.3e$, $E_{FEM}=%.3e$  |  ', ...
                     '$I_0=%d$, $T_0=%.2fs$, ', ...
                     '$I_p=%d$, $T_p=%.2fs$'], ...
                     case_name, k, ...
                     err_model_vs_fem, err_fem_predinit_vs_zero, ...
                     iter_zero, time_zero, iter_pred, time_pred), ...
        'Interpreter', 'latex', ...
        'FontName', style.fontName, ...
        'FontSize', style.sgTitleFontSize, ...
        'FontWeight', 'bold');

end


function A = close_theta(A)
    A = [A, A(:, 1)];
end


function plot_field_sci(X, Y, Z, cmin, cmax, ttl, style)

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
        'FontName', style.fontName, ...
        'FontSize', style.axisFontSize, ...
        'LineWidth', style.lineWidth, ...
        'TickDir', 'out', ...
        'Layer', 'top', ...
        'XGrid', 'off', ...
        'YGrid', 'off', ...
        'ZGrid', 'off');

    caxis([cmin, cmax]);

    cb = colorbar;
    cb.FontSize = style.cbFontSize;
    cb.TickDirection = 'out';

    title(ttl, ...
        'Interpreter', 'latex', ...
        'FontName', style.fontName, ...
        'FontSize', style.titleFontSize);

    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', style.axisFontSize);
    ylabel('$y$', 'Interpreter', 'latex', 'FontSize', style.axisFontSize);

end


