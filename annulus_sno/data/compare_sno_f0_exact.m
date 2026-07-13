clear; clc; close all;

%% ============================================================
%  1. Load SNO prediction
% =============================================================
mat_file = 'sno_f0_pod_prediction_physv2.mat';
data = load(mat_file);

pod_coords = data.pod_coords;       % [Npod, 2]
u_sno = data.u_sno_pod(:);          % [Npod, 1]

k = data.k_value;
a = data.r_inner;
R = data.r_outer;

theta_size = data.theta_size;
radial_size = data.radial_size;

x = pod_coords(:, 1);
y = pod_coords(:, 2);

r = sqrt(x.^2 + y.^2);
theta = atan2(y, x);

fprintf('Loaded file: %s\n', mat_file);
fprintf('r_inner = %.8g, r_outer = %.8g, k = %.8g\n', a, R, k);
fprintf('Npod = %d\n', numel(u_sno));

%% ============================================================
%  2. Compute analytical solution on the same pod grid
% =============================================================
ka = k * a;
kR = k * R;

% Modified Bessel functions
I1_R = besseli(1, kR);
K1_R = besselk(1, kR);

I1p_a = 0.5 * (besseli(0, ka) + besseli(2, ka));
K1p_a = -0.5 * (besselk(0, ka) + besselk(2, ka));

% Solve for A, B:
% A*I1(kR) + B*K1(kR) = 0
% k*(A*I1'(ka) + B*K1'(ka)) = -1
M = [I1_R, K1_R;
     k * I1p_a, k * K1p_a];

rhs = [0; -1];

coef = M \ rhs;
A = coef(1);
B = coef(2);

fprintf('A = %.16e\n', A);
fprintf('B = %.16e\n', B);

% u_exact = (A * besseli(1, k * r) + B * besselk(1, k * r)) .* cos(theta);
u_exact = 0.0384615 * (1./r - r) .* cos(theta);

%% ============================================================
%  3. Relative L2 error
% =============================================================
abs_err = abs(u_sno - u_exact);

rel_l2 = norm(u_sno - u_exact, 2) / max(norm(u_exact, 2), eps);
rel_linf = norm(u_sno - u_exact, inf) / max(norm(u_exact, inf), eps);

fprintf('Relative L2 error   = %.8e\n', rel_l2);
fprintf('Relative Linf error = %.8e\n', rel_linf);

%% ============================================================
%  4. Publication-quality figure
% =============================================================
fig_dir = 'fig_sno_f0_exact';
if ~exist(fig_dir, 'dir')
    mkdir(fig_dir);
end

% Global style
set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultTextFontSize', 12);
set(groot, 'defaultAxesLineWidth', 1.0);
set(groot, 'defaultLineLineWidth', 1.2);

% Same color scale for SNO and exact
cmin = min([u_sno; u_exact]);
cmax = max([u_sno; u_exact]);

% Error color scale
emin = 0;
emax = max(abs_err);
if emax <= 0
    emax = 1e-14;
end

fig = figure('Units', 'centimeters', ...
             'Position', [3, 3, 25, 7.5], ...
             'Color', 'w');

tiledlayout(1, 3, ...
    'Padding', 'compact', ...
    'TileSpacing', 'compact');

%% ---- Subplot 1: SNO prediction ----
nexttile;
scatter(x, y, 16, u_sno, 'filled', ...
    'MarkerEdgeColor', 'none');
axis equal tight;
box on;
colormap(gca, turbo);
caxis([cmin, cmax]);
cb = colorbar;
cb.Label.String = '$u$';
cb.Label.Interpreter = 'latex';
title('$u_{\mathrm{SNO}}$', 'Interpreter', 'latex');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$y$', 'Interpreter', 'latex');
set(gca, 'TickDir', 'out', 'Layer', 'top');

%% ---- Subplot 2: analytical solution ----
nexttile;
scatter(x, y, 16, u_exact, 'filled', ...
    'MarkerEdgeColor', 'none');
axis equal tight;
box on;
colormap(gca, turbo);
caxis([cmin, cmax]);
cb = colorbar;
cb.Label.String = '$u$';
cb.Label.Interpreter = 'latex';
title('$u_{\mathrm{exact}}$', 'Interpreter', 'latex');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$y$', 'Interpreter', 'latex');
set(gca, 'TickDir', 'out', 'Layer', 'top');

%% ---- Subplot 3: absolute error ----
nexttile;
scatter(x, y, 16, abs_err, 'filled', ...
    'MarkerEdgeColor', 'none');
axis equal tight;
box on;
colormap(gca, hot);
caxis([emin, emax]);
cb = colorbar;
cb.Label.String = '$|u_{\mathrm{SNO}}-u_{\mathrm{exact}}|$';
cb.Label.Interpreter = 'latex';
title('$|u_{\mathrm{SNO}}-u_{\mathrm{exact}}|$', 'Interpreter', 'latex');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$y$', 'Interpreter', 'latex');
set(gca, 'TickDir', 'out', 'Layer', 'top');

%% ---- Global title ----
sgtitle(sprintf('$f=0$ benchmark: relative $L^2$ error = %.3e', rel_l2), ...
    'Interpreter', 'latex', ...
    'FontSize', 14, ...
    'FontWeight', 'normal');

%% ============================================================
%  5. Save figure and exact solution
% =============================================================
% png_name = fullfile(fig_dir, 'sno_vs_exact_f0_pod.png');
% pdf_name = fullfile(fig_dir, 'sno_vs_exact_f0_pod.pdf');
% mat_out = fullfile(fig_dir, 'sno_vs_exact_f0_pod_results.mat');
% 
% exportgraphics(fig, png_name, 'Resolution', 400);
% exportgraphics(fig, pdf_name, 'ContentType', 'vector');
% 
% save(mat_out, ...
%     'pod_coords', ...
%     'u_sno', ...
%     'u_exact', ...
%     'abs_err', ...
%     'rel_l2', ...
%     'rel_linf', ...
%     'A', ...
%     'B', ...
%     'k', ...
%     'a', ...
%     'R');
% 
% fprintf('Figure saved to:\n%s\n%s\n', png_name, pdf_name);
% fprintf('Results saved to:\n%s\n', mat_out);