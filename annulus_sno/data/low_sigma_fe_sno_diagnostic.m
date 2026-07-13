clear; clc; close all;

%% ============================================================
%  User settings
%% ============================================================

mat_file = 'low_sigma_fe_sno_diagnostic.mat';

% 选择要看的 sigma
selected_sigma = 2.0;     % 可改为 0.5, 1.0, 2.0

% 选择样本 id 范围
id_range = 1:2;          % 例如 1:5, 20:30

% 图题误差采用 pod 还是 probe
% 推荐：
%   'pod'   : 与当前规则网格可视化对应
%   'probe' : 与随机点泛化误差对应
error_metric_for_title = 'pod';

% 是否对误差图使用统一色阶
use_shared_error_caxis = true;

% 是否对三个解图使用统一色阶
use_shared_solution_caxis = true;

% 图像风格
font_name = 'Times New Roman';
font_size = 12;
marker_edge = 'none';

%% ============================================================
%  Load data
%% ============================================================

S = load(mat_file);

required_vars = {'sigma_values', 'x_grid', 'y_grid'};
for i = 1:numel(required_vars)
    if ~isfield(S, required_vars{i})
        error('Missing variable "%s" in mat file.', required_vars{i});
    end
end

sigma_values = double(S.sigma_values(:));

[~, i_sigma] = min(abs(sigma_values - selected_sigma));
sigma_actual = sigma_values(i_sigma);

if abs(sigma_actual - selected_sigma) > 1e-10
    warning('Selected sigma %.6g not found exactly. Use nearest sigma %.6g.', ...
        selected_sigma, sigma_actual);
end

% 坐标网格
X = double(S.x_grid);
Y = double(S.y_grid);

% 保证 X/Y 是二维矩阵
X = squeeze(X);
Y = squeeze(Y);

[Nr, Nt] = size(X);

% 检查样本数量
n_sample = infer_num_samples(S, i_sigma);

id_range = id_range(id_range >= 1 & id_range <= n_sample);
if isempty(id_range)
    error('id_range is empty after checking valid sample range. n_sample = %d.', n_sample);
end

fprintf('Loaded: %s\n', mat_file);
fprintf('sigma selected = %.6g, sigma actual = %.6g, sigma index = %d\n', ...
    selected_sigma, sigma_actual, i_sigma);
fprintf('Valid sample id range: %d to %d\n', min(id_range), max(id_range));

%% ============================================================
%  Global plotting style
%% ============================================================

set(groot, 'defaultFigureColor', 'w');
set(groot, 'defaultAxesFontName', font_name);
set(groot, 'defaultAxesFontSize', font_size);
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

%% ============================================================
%  Main visualization loop
%% ============================================================

for idx = id_range

    %% ------------------------------------------------------------
    %  Extract fields
    %% ------------------------------------------------------------

    U_true = get_grid_field(S, 'u_true', i_sigma, idx, Nr, Nt);
    U_fe   = get_grid_field(S, 'u_fe',   i_sigma, idx, Nr, Nt);
    U_sno  = get_grid_field(S, 'u_sno',  i_sigma, idx, Nr, Nt);

    E_fe  = abs(U_fe  - U_true);
    E_sno = abs(U_sno - U_true);

    %% ------------------------------------------------------------
    %  Extract relative L2 errors
    %% ------------------------------------------------------------

    switch lower(error_metric_for_title)
        case 'pod'
            err_fe_l2  = get_error_value(S, 'err_fe_u_pod',  i_sigma, idx, U_fe,  U_true);
            err_sno_l2 = get_error_value(S, 'err_sno_u_pod', i_sigma, idx, U_sno, U_true);

        case 'probe'
            err_fe_l2  = get_error_value(S, 'err_fe_u_probe',  i_sigma, idx, U_fe,  U_true);
            err_sno_l2 = get_error_value(S, 'err_sno_u_probe', i_sigma, idx, U_sno, U_true);

        otherwise
            error('error_metric_for_title must be "pod" or "probe".');
    end

    %% ------------------------------------------------------------
    %  Close theta direction for smooth annulus visualization
    %% ------------------------------------------------------------

    Xc = close_theta(X);
    Yc = close_theta(Y);

    U_true_c = close_theta(U_true);
    U_fe_c   = close_theta(U_fe);
    U_sno_c  = close_theta(U_sno);

    E_fe_c   = close_theta(E_fe);
    E_sno_c  = close_theta(E_sno);

    %% ------------------------------------------------------------
    %  Color axis
    %% ------------------------------------------------------------

    if use_shared_solution_caxis
        u_min = min([U_true_c(:); U_fe_c(:); U_sno_c(:)]);
        u_max = max([U_true_c(:); U_fe_c(:); U_sno_c(:)]);

        if abs(u_max - u_min) < eps
            u_min = u_min - 1;
            u_max = u_max + 1;
        end
    end

    if use_shared_error_caxis
        e_min = 0.0;
        e_max = max([E_fe_c(:); E_sno_c(:)]);

        if e_max <= 0 || ~isfinite(e_max)
            e_max = eps;
        end
    end

    %% ------------------------------------------------------------
    %  Figure layout: 2 rows × 6 columns
    %  Top row: 3 panels, each spans 2 columns
    %  Bottom row: 2 panels centered, each spans 2 columns
    %% ------------------------------------------------------------

    fig = figure( ...
        'Color', 'w', ...
        'Units', 'centimeters', ...
        'Position', [3, 3, 30, 16]);

    tl = tiledlayout(fig, 2, 6, ...
        'TileSpacing', 'compact', ...
        'Padding', 'compact');

    %% -------------------- True solution --------------------
    ax1 = nexttile(tl, 1, [1, 2]);
    plot_annulus_field(ax1, Xc, Yc, U_true_c, '$u_{\mathrm{true}}$', marker_edge);
    colormap(ax1, parula(256));
    if use_shared_solution_caxis
        caxis(ax1, [u_min, u_max]);
    end
    cb1 = colorbar(ax1);
    cb1.TickLabelInterpreter = 'latex';

    %% -------------------- FE reconstruction --------------------
    ax2 = nexttile(tl, 3, [1, 2]);
    plot_annulus_field(ax2, Xc, Yc, U_fe_c, '$u_{\mathrm{FE}}$', marker_edge);
    colormap(ax2, parula(256));
    if use_shared_solution_caxis
        caxis(ax2, [u_min, u_max]);
    end
    cb2 = colorbar(ax2);
    cb2.TickLabelInterpreter = 'latex';

    %% -------------------- SNO prediction --------------------
    ax3 = nexttile(tl, 5, [1, 2]);
    plot_annulus_field(ax3, Xc, Yc, U_sno_c, '$u_{\mathrm{SNO}}$', marker_edge);
    colormap(ax3, parula(256));
    if use_shared_solution_caxis
        caxis(ax3, [u_min, u_max]);
    end
    cb3 = colorbar(ax3);
    cb3.TickLabelInterpreter = 'latex';

    %% -------------------- FE absolute error --------------------
    ax4 = nexttile(tl, 8, [1, 2]);
    plot_annulus_field(ax4, Xc, Yc, E_fe_c, '$|u_{\mathrm{FE}}-u_{\mathrm{true}}|$', marker_edge);
    colormap(ax4, hot(256));
    if use_shared_error_caxis
        caxis(ax4, [e_min, e_max]);
    else
        caxis(ax4, [0, max(E_fe_c(:)) + eps]);
    end
    cb4 = colorbar(ax4);
    cb4.TickLabelInterpreter = 'latex';

    %% -------------------- SNO absolute error --------------------
    ax5 = nexttile(tl, 10, [1, 2]);
    plot_annulus_field(ax5, Xc, Yc, E_sno_c, '$|u_{\mathrm{SNO}}-u_{\mathrm{true}}|$', marker_edge);
    colormap(ax5, hot(256));
    if use_shared_error_caxis
        caxis(ax5, [e_min, e_max]);
    else
        caxis(ax5, [0, max(E_sno_c(:)) + eps]);
    end
    cb5 = colorbar(ax5);
    cb5.TickLabelInterpreter = 'latex';

    %% ------------------------------------------------------------
    %  Super title
    %% ------------------------------------------------------------

    sgtitle(tl, sprintf(['$\\sigma=%.2f$, sample id = %d, ', ...
                         'FE $L^2_{rel}$ = %.3e, SNO $L^2_{rel}$ = %.3e'], ...
                         sigma_actual, idx, err_fe_l2, err_sno_l2), ...
        'Interpreter', 'latex', ...
        'FontSize', 15, ...
        'FontWeight', 'bold');

    drawnow;

end

%% ============================================================
%  Local functions
%% ============================================================

function Zc = close_theta(Z)
    % Close periodic theta direction by appending the first column.
    Z = squeeze(double(Z));
    Zc = [Z, Z(:, 1)];
end


function plot_annulus_field(ax, X, Y, Z, ttl, marker_edge)
    axes(ax);

    h = surf(ax, X, Y, Z);

    set(h, ...
        'EdgeColor', marker_edge, ...
        'LineStyle', 'none', ...
        'FaceColor', 'interp');

    view(ax, 2);
    axis(ax, 'equal');
    axis(ax, 'tight');
    axis(ax, 'off');
    box(ax, 'off');

    title(ax, ttl, ...
        'Interpreter', 'latex', ...
        'FontSize', 13, ...
        'FontWeight', 'normal');
end


function n_sample = infer_num_samples(S, i_sigma)
    % Infer number of samples from available grid or pod variables.

    candidate_names = { ...
        'u_true_grid', ...
        'u_true_pod', ...
        'u_sno_grid', ...
        'u_sno_pod'};

    n_sample = [];

    for k = 1:numel(candidate_names)
        name = candidate_names{k};

        if isfield(S, name)
            A = S.(name);
            sz = size(A);

            if numel(sz) >= 2
                n_sample = sz(2);
                break;
            end
        end
    end

    if isempty(n_sample)
        error('Cannot infer number of samples from mat file.');
    end

    if i_sigma > size(S.sigma_values, 1) && i_sigma > numel(S.sigma_values)
        error('Invalid sigma index.');
    end
end


function Z = get_grid_field(S, field_key, i_sigma, idx, Nr, Nt)
    % Robustly read grid field.
    %
    % Supported field_key:
    %   'u_true'
    %   'u_fe'
    %   'u_sno'
    %
    % Preferred variables:
    %   u_true_grid, u_fe_grid, u_sno_grid
    %
    % Fallback variables:
    %   u_true_pod, u_fe_pod, u_sno_pod

    switch field_key
        case 'u_true'
            grid_name = 'u_true_grid';
            pod_name  = 'u_true_pod';

        case 'u_fe'
            grid_name = 'u_fe_grid';
            pod_name  = 'u_fe_pod';

        case 'u_sno'
            grid_name = 'u_sno_grid';
            pod_name  = 'u_sno_pod';

        otherwise
            error('Unknown field_key: %s', field_key);
    end

    if isfield(S, grid_name)
        A = S.(grid_name);
        Z = squeeze(double(A(i_sigma, idx, :, :)));

    elseif isfield(S, pod_name)
        A = S.(pod_name);
        v = squeeze(double(A(i_sigma, idx, :)));
        Z = reshape(v, Nr, Nt);

    else
        error('Cannot find "%s" or "%s" in mat file.', grid_name, pod_name);
    end

    if ~isequal(size(Z), [Nr, Nt])
        Z = reshape(Z, Nr, Nt);
    end
end


function err = get_error_value(S, err_name, i_sigma, idx, U_pred, U_true)
    % Read relative L2 error from mat file.
    % If the error variable is missing, compute it from current grid fields.

    if isfield(S, err_name)
        E = S.(err_name);
        err = double(squeeze(E(i_sigma, idx)));

    else
        numerator = norm(U_pred(:) - U_true(:), 2);
        denominator = norm(U_true(:), 2) + eps;
        err = numerator / denominator;

        warning('Variable "%s" not found. Error is computed from grid fields.', err_name);
    end
end