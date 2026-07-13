clear; clc; close all;

%% ============================================================
%  1. Load SNO prediction
% =============================================================
mat_file = 'sno_f0_pod_prediction.mat';
data = load(mat_file);

pod_coords = data.pod_coords;       % [Npod, 2]
u_sno = data.u_sno_pod(:);          % [Npod, 1]

k = double(data.k_value);
rin = double(data.r_inner);
rout = double(data.r_outer);

x = pod_coords(:, 1);
y = pod_coords(:, 2);

r = sqrt(x.^2 + y.^2);
theta = atan2(y, x);

fprintf('Loaded file: %s\n', mat_file);
fprintf('rin = %.12g, rout = %.12g, k = %.12g\n', rin, rout, k);
fprintf('Npod = %d\n', numel(u_sno));

%% ============================================================
%  2. Solve radial BVP numerically
%
%     R'' + (1/r)R' - (1/r^2)R - k^2 R = 0
%
%     BC:
%         R(rout) = 0
%         R'(rin) = -1
% =============================================================

% Initial mesh
Nr_bvp = 300;
rmesh = linspace(rin, rout, Nr_bvp);

% Good initial guess from k -> 0 Laplace limit:
% R(r) = C*r + D/r
C0 = -rin^2 / (rin^2 + rout^2);
D0 =  rin^2 * rout^2 / (rin^2 + rout^2);

guess_fun = @(rr) [ ...
    C0 * rr + D0 ./ rr; ...
    C0 - D0 ./ rr.^2 ...
];

solinit = bvpinit(rmesh, guess_fun);

% Solver options
opts = bvpset( ...
    'RelTol', 1e-11, ...
    'AbsTol', 1e-13, ...
    'NMax', 50000, ...
    'Stats', 'on');

% Use bvp5c if available; otherwise fallback to bvp4c
if exist('bvp5c', 'file') == 2
    fprintf('Using bvp5c...\n');
    sol = bvp5c(@(rr, yy) radial_ode(rr, yy, k), ...
                @(ya, yb) radial_bc(ya, yb), ...
                solinit, opts);
else
    fprintf('bvp5c not found. Using bvp4c...\n');
    sol = bvp4c(@(rr, yy) radial_ode(rr, yy, k), ...
                @(ya, yb) radial_bc(ya, yb), ...
                solinit, opts);
end

%% ============================================================
%  3. Evaluate BVP numerical solution on SNO pod grid
% =============================================================
R_eval = deval(sol, r);
R_num = R_eval(1, :).';

u_bvp = R_num .* cos(theta);

%% ============================================================
%  4. Error metrics
% =============================================================
abs_err = abs(u_sno - u_bvp);

rel_l2 = norm(u_sno - u_bvp, 2) / max(norm(u_bvp, 2), eps);
rel_linf = norm(u_sno - u_bvp, inf) / max(norm(u_bvp, inf), eps);

fprintf('\n========== Error between SNO and BVP numerical solution ==========\n');
fprintf('Relative L2 error   = %.8e\n', rel_l2);
fprintf('Relative Linf error = %.8e\n', rel_linf);
fprintf('Max absolute error  = %.8e\n', max(abs_err));
fprintf('Mean absolute error = %.8e\n', mean(abs_err));

%% ============================================================
%  5. Publication-quality figure
% =============================================================
fig_dir = 'fig_sno_f0_bvp';
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

% Same color scale for SNO and BVP
cmin = min([u_sno; u_bvp]);
cmax = max([u_sno; u_bvp]);

% Error color scale
emin = 0;
emax = max(abs_err);
if emax <= 0
    emax = 1e-14;
end

fig = figure('Units', 'centimeters', ...
             'Position', [3, 3, 25, 7.5], ...
             'Color', 'w');

%% ---- Subplot 1: SNO prediction ----
subplot(1, 3, 1);
scatter(x, y, 16, u_sno, 'filled', ...
    'MarkerEdgeColor', 'none');
axis equal tight;
box on;
colormap(gca, parula);
caxis([cmin, cmax]);
cb = colorbar;
cb.Label.String = '$u$';
cb.Label.Interpreter = 'latex';
title('$u_{\mathrm{SNO}}$', 'Interpreter', 'latex');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$y$', 'Interpreter', 'latex');
set(gca, 'TickDir', 'out', 'Layer', 'top');

%% ---- Subplot 2: BVP numerical solution ----
subplot(1, 3, 2);
scatter(x, y, 16, u_bvp, 'filled', ...
    'MarkerEdgeColor', 'none');
axis equal tight;
box on;
colormap(gca, parula);
caxis([cmin, cmax]);
cb = colorbar;
cb.Label.String = '$u$';
cb.Label.Interpreter = 'latex';
title('$u_{\mathrm{BVP}}$', 'Interpreter', 'latex');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$y$', 'Interpreter', 'latex');
set(gca, 'TickDir', 'out', 'Layer', 'top');

%% ---- Subplot 3: absolute error ----
subplot(1, 3, 3);
scatter(x, y, 16, abs_err, 'filled', ...
    'MarkerEdgeColor', 'none');
axis equal tight;
box on;
colormap(gca, hot);
caxis([emin, emax]);
cb = colorbar;
cb.Label.String = '$|u_{\mathrm{SNO}}-u_{\mathrm{BVP}}|$';
cb.Label.Interpreter = 'latex';
title('$|u_{\mathrm{SNO}}-u_{\mathrm{BVP}}|$', 'Interpreter', 'latex');
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$y$', 'Interpreter', 'latex');
set(gca, 'TickDir', 'out', 'Layer', 'top');

% Overall title
sgtitle(sprintf('$f=0$ benchmark: SNO vs. BVP, relative $L^2$ error = %.3e', rel_l2), ...
    'Interpreter', 'latex', ...
    'FontSize', 14, ...
    'FontWeight', 'normal');

%% ============================================================
%  6. Save figure and results
% =============================================================
% png_name = fullfile(fig_dir, 'sno_vs_bvp_f0_pod.png');
% pdf_name = fullfile(fig_dir, 'sno_vs_bvp_f0_pod.pdf');
% mat_out = fullfile(fig_dir, 'sno_vs_bvp_f0_pod_results.mat');
% 
% print(fig, png_name, '-dpng', '-r400');
% print(fig, pdf_name, '-dpdf', '-painters');
% 
% save(mat_out, ...
%     'pod_coords', ...
%     'u_sno', ...
%     'u_bvp', ...
%     'abs_err', ...
%     'rel_l2', ...
%     'rel_linf', ...
%     'k', ...
%     'rin', ...
%     'rout', ...
%     'sol');
% 
% fprintf('\nFigure saved to:\n%s\n%s\n', png_name, pdf_name);
% fprintf('Results saved to:\n%s\n', mat_out);

%% ============================================================
%  Local functions
% =============================================================
function dydr = radial_ode(rr, yy, k)
    % yy(1,:) = R(r)
    % yy(2,:) = R'(r)
    %
    % R'' + (1/r)R' - (1/r^2)R - k^2 R = 0
    %
    % Therefore:
    % R'' = -(1/r)R' + (1/r^2 + k^2)R

    R = yy(1, :);
    Rp = yy(2, :);

    dydr = zeros(size(yy));
    dydr(1, :) = Rp;
    dydr(2, :) = -(1 ./ rr) .* Rp + (1 ./ rr.^2 + k^2) .* R;
end

function res = radial_bc(ya, yb)
    % ya = y(rin), yb = y(rout)
    %
    % Inner Neumann:
    %     R'(rin) = -1
    %
    % Outer Dirichlet:
    %     R(rout) = 0

    res = [
        ya(2) + 1;   % R'(rin) + 1 = 0
        yb(1)        % R(rout) = 0
    ];
end