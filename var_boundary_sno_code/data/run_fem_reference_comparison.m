clear; clc; close all;

%% ============================================================
% Select case
%% ============================================================

% -------- Circle --------
% case_file = 'circle_f0_flux_cos_theta_transformer_fe_k_0p5_1p0_1p5_fem_refgrid.mat'; %'circle_f0_flux_cos_theta_transformer_fe_k_0p5_1p0_1p5.mat';
% geom.type = 'circle';
% geom.r_inner = 0.2;
% geom.outer_scale = 5.0;

% % -------- Square --------
% case_file = 'square_corner_radius_0p2_f0_problem_flux_transformer_fe_k_0p5_1p0_1p5.mat'; %'square_corner_radius_0p2_f0_problem_flux_transformer_fe_k_0p5_1p0_1p5.mat';
% geom.type = 'square';
% geom.corner_radius = 0.2;
% geom.outer_scale = 5.0;

% -------- Pentagon --------
case_file = 'pentagon_corner_radius_0p2_f0_problem_flux_transformer_fe_k_0p5_1p0_1p5.mat'; %'pentagon_corner_radius_0p2_f0_problem_flux_transformer_fe_k_0p5_1p0_1p5.mat';
geom.type = 'polygon';
geom.n_sides = 5;
geom.corner_radius = 0.2;
geom.rotation = pi/10;   % 若一个角点在 +y 轴
geom.outer_scale = 5.0;

S = load(case_file);

k_values = S.k_values(:);
num_cases = numel(k_values);

%% ============================================================
% Target grid from neural prediction
%% ============================================================

if isfield(S, 'x_grid')
    Xtar = double(S.x_grid);
    Ytar = double(S.y_grid);
elseif isfield(S, 'x_phys_grid')
    Xtar = double(S.x_phys_grid);
    Ytar = double(S.y_phys_grid);
else
    error('Cannot find target coordinate grid in mat file.');
end

U_pred_all = double(S.u_pred_grid);

[Nr_tar, Nt_tar] = size(Xtar);

%% ============================================================
% FEM reference mesh resolution
% Use finer mesh than neural grid for reference accuracy.
%% ============================================================

Nr_ref = max(120, 16 * Nr_tar);
Nt_ref = max(512, 32 * Nt_tar);

% Nr_ref = double(S.Nr_ref);
% Nt_ref = double(S.Nt_ref);

fprintf('Reference FEM mesh: Nr=%d, Nt=%d\n', Nr_ref, Nt_ref);

%% ============================================================
% Solve each k case
%% ============================================================

U_ref_on_target_all = zeros(num_cases, Nr_tar, Nt_tar);
Abs_err_all = zeros(num_cases, Nr_tar, Nt_tar);
Rel_err_all = zeros(num_cases, 1);

for icase = 1:num_cases

    k = k_values(icase);

    fprintf('\nSolving FEM reference for k = %.4f\n', k);
    
    fem = solve_annulus_star_fem(geom, k, Nr_ref, Nt_ref);

    % Interpolate FEM solution to neural prediction grid
    Finterp = scatteredInterpolant( ...
        double(fem.x(:)), ...
        double(fem.y(:)), ...
        double(fem.u_grid(:)), ...
        'linear', ...
        'nearest');

    U_ref_tar = Finterp(Xtar, Ytar);

    U_pred = squeeze(U_pred_all(icase, :, :));

    Abs_err = abs(U_pred - U_ref_tar);

    Rel_err = norm(U_pred(:) - U_ref_tar(:), 2) / ...
              (norm(U_ref_tar(:), 2) + eps);

    U_ref_on_target_all(icase, :, :) = U_ref_tar;
    Abs_err_all(icase, :, :) = Abs_err;
    Rel_err_all(icase) = Rel_err;

    fprintf('Relative error vs FEM: %.6e\n', Rel_err);

end

%% ============================================================
% Quick visualization
%% ============================================================

for icase = 1:num_cases

    k = k_values(icase);

    U_pred = squeeze(U_pred_all(icase, :, :));
    U_ref  = squeeze(U_ref_on_target_all(icase, :, :));
    U_err  = squeeze(Abs_err_all(icase, :, :));

    X = close_theta(Xtar);
    Y = close_theta(Ytar);

    U_pred_c = close_theta(U_pred);
    U_ref_c  = close_theta(U_ref);
    U_err_c  = close_theta(U_err);

    u_min = min([U_pred_c(:); U_ref_c(:)]);
    u_max = max([U_pred_c(:); U_ref_c(:)]);

    err_max = max(U_err_c(:));
    if err_max <= 0
        err_max = eps;
    end

    figure('Color', 'w', ...
           'Units', 'centimeters', ...
           'Position', [3, 3, 28, 8.5]);

    tiledlayout(1, 3, ...
        'TileSpacing', 'compact', ...
        'Padding', 'loose');

    colormap(turbo);

    nexttile;
    plot_field(X, Y, U_pred_c, u_min, u_max, '$u_{\mathrm{pred}}$');

    nexttile;
    plot_field(X, Y, U_ref_c, u_min, u_max, '$u_{\mathrm{FEM}}$');

    nexttile;
    plot_field(X, Y, U_err_c, 0, err_max, '$|u_{\mathrm{pred}}-u_{\mathrm{FEM}}|$');

    sgtitle(sprintf('$k=%.2f$, relative error = %.3e', ...
        k, Rel_err_all(icase)), ...
        'Interpreter', 'latex', ...
        'FontSize', 14, ...
        'FontWeight', 'bold');

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