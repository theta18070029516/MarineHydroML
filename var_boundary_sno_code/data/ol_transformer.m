clear; clc; close all;

%% ============================================================
% Load data
%% ============================================================
data_path = 'ol_transformer_fe_test_100_varboundary.mat';
S = load(data_path);

%% ============================================================
% Select sample index
%% ============================================================
for idx = 88   % 修改这里查看不同样本，例如 idx = 10

%% ============================================================
% Read physical coordinates
%% ============================================================
X = squeeze(S.x_phys_grid(idx, :, :));
Y = squeeze(S.y_phys_grid(idx, :, :));

%% ============================================================
% Read fields
% u_pred  : Transformer + FE prediction
% u_recon : FE oracle reconstruction
% u_true  : ground truth
%% ============================================================
U_pred = squeeze(S.u_pred_grid(idx, :, :));

% 兼容两种保存命名：
% 新的 Transformer 测试文件通常是 u_fe_recon_grid；
% 如果你旧文件里叫 u_recon_grid，也可以自动识别。
if isfield(S, 'u_fe_recon_grid')
    U_recon = squeeze(S.u_fe_recon_grid(idx, :, :));
elseif isfield(S, 'u_recon_grid')
    U_recon = squeeze(S.u_recon_grid(idx, :, :));
else
    error('Cannot find u reconstruction field. Expected u_fe_recon_grid or u_recon_grid.');
end

U_true = squeeze(S.u_true_grid(idx, :, :));

%% ============================================================
% Errors
%% ============================================================
Err_pred_recon = abs(U_pred  - U_recon);
Err_recon_true = abs(U_recon - U_true);
Err_pred_true  = abs(U_pred  - U_true);

%% ============================================================
% Close periodic theta direction
% All grid fields are [Nr, Nt]. Append the first theta column.
%% ============================================================
X = close_theta(X);
Y = close_theta(Y);

U_pred  = close_theta(U_pred);
U_recon = close_theta(U_recon);
U_true  = close_theta(U_true);

Err_pred_recon = close_theta(Err_pred_recon);
Err_recon_true = close_theta(Err_recon_true);
Err_pred_true  = close_theta(Err_pred_true);

%% ============================================================
% Boundary curve, also closed
%% ============================================================
has_boundary = isfield(S, 'boundary_coords');

if has_boundary
    xb = squeeze(S.boundary_coords(idx, :, 1));
    yb = squeeze(S.boundary_coords(idx, :, 2));

    xb = [xb(:); xb(1)];
    yb = [yb(:); yb(1)];
end

%% ============================================================
% Relative errors and k
%% ============================================================
if isfield(S, 'err_u_pred_each')
    err_pred = S.err_u_pred_each(idx);
else
    err_pred = norm(U_pred(:) - U_true(:), 2) / (norm(U_true(:), 2) + eps);
end

if isfield(S, 'err_u_fe_each')
    err_recon = S.err_u_fe_each(idx);
else
    err_recon = norm(U_recon(:) - U_true(:), 2) / (norm(U_true(:), 2) + eps);
end

if isfield(S, 'err_latent_each')
    err_latent = S.err_latent_each(idx);
else
    err_latent = NaN;
end

if isfield(S, 'k_values')
    k_val = S.k_values(idx, 1);
else
    k_val = NaN;
end

%% ============================================================
% Color limits
%% ============================================================
% 第一行三个 u 场使用同一个色标，便于公平对比
u_min = min([U_pred(:); U_recon(:); U_true(:)]);
u_max = max([U_pred(:); U_recon(:); U_true(:)]);

% 第二行三个误差图使用同一个色标，便于比较误差来源
err_max = max([Err_pred_recon(:); Err_recon_true(:); Err_pred_true(:)]);
if err_max <= 0
    err_max = eps;
end

%% ============================================================
% Figure settings: publication style
%% ============================================================
fig = figure('Color', 'w', ...
             'Units', 'centimeters', ...
             'Position', [2, 2, 32, 17]);

tl = tiledlayout(2, 3, ...
    'TileSpacing', 'compact', ...
    'Padding', 'loose');

fontName = 'Arial';
axisFontSize = 9.5;
titleFontSize = 11;
sgTitleFontSize = 13;
cbFontSize = 8.5;
lineWidth = 0.85;

colormap(turbo);

%% ============================================================
% Row 1: u_pred, u_recon, u_true
%% ============================================================

nexttile(1);
plot_field(X, Y, U_pred, u_min, u_max, ...
    '$u_{\mathrm{pred}}$', fontName, axisFontSize, titleFontSize, cbFontSize, lineWidth);
if has_boundary
    hold on; plot(xb, yb, 'r-', 'LineWidth', 1.0); hold off;
end

nexttile(2);
plot_field(X, Y, U_recon, u_min, u_max, ...
    '$u_{\mathrm{recon}}$', fontName, axisFontSize, titleFontSize, cbFontSize, lineWidth);
if has_boundary
    hold on; plot(xb, yb, 'r-', 'LineWidth', 1.0); hold off;
end

nexttile(3);
plot_field(X, Y, U_true, u_min, u_max, ...
    '$u_{\mathrm{true}}$', fontName, axisFontSize, titleFontSize, cbFontSize, lineWidth);
if has_boundary
    hold on; plot(xb, yb, 'r-', 'LineWidth', 1.0); hold off;
end

%% ============================================================
% Row 2: absolute errors
%% ============================================================

nexttile(4);
plot_field(X, Y, Err_pred_recon, 0, err_max, ...
    '$|u_{\mathrm{pred}}-u_{\mathrm{recon}}|$', fontName, axisFontSize, titleFontSize, cbFontSize, lineWidth);
if has_boundary
    hold on; plot(xb, yb, 'r-', 'LineWidth', 1.0); hold off;
end

nexttile(5);
plot_field(X, Y, Err_recon_true, 0, err_max, ...
    '$|u_{\mathrm{recon}}-u_{\mathrm{true}}|$', fontName, axisFontSize, titleFontSize, cbFontSize, lineWidth);
if has_boundary
    hold on; plot(xb, yb, 'r-', 'LineWidth', 1.0); hold off;
end

nexttile(6);
plot_field(X, Y, Err_pred_true, 0, err_max, ...
    '$|u_{\mathrm{pred}}-u_{\mathrm{true}}|$', fontName, axisFontSize, titleFontSize, cbFontSize, lineWidth);
if has_boundary
    hold on; plot(xb, yb, 'r-', 'LineWidth', 1.0); hold off;
end

%% ============================================================
% Global title
%% ============================================================
if isnan(err_latent)
    sgtitle(sprintf(['Transformer--FE prediction, sample %d  |  ', ...
        '$k=%.4f$,  $E_{u}^{pred}=%.3e$,  $E_{u}^{FE}=%.3e$'], ...
        idx, k_val, err_pred, err_recon), ...
        'Interpreter', 'latex', ...
        'FontName', fontName, ...
        'FontSize', sgTitleFontSize, ...
        'FontWeight', 'bold');
else
    sgtitle(sprintf(['Transformer--FE prediction, sample %d  |  ', ...
        '$k=%.4f$,  $E_{z}=%.3e$,  $E_{u}^{pred}=%.3e$,  $E_{u}^{FE}=%.3e$'], ...
        idx, k_val, err_latent, err_pred, err_recon), ...
        'Interpreter', 'latex', ...
        'FontName', fontName, ...
        'FontSize', sgTitleFontSize, ...
        'FontWeight', 'bold');
end

% %% ============================================================
% % Save figure
% %% ============================================================
% save_dir = './figures_ol_transformer_fe';
% if ~exist(save_dir, 'dir')
%     mkdir(save_dir);
% end
% 
% png_path = fullfile(save_dir, sprintf('OL_FE_sample_%03d.png', idx));
% pdf_path = fullfile(save_dir, sprintf('OL_FE_sample_%03d.pdf', idx));
% 
% % 如果你的 MATLAB 支持 exportgraphics，优先使用
% if exist('exportgraphics', 'file')
%     exportgraphics(tl, png_path, 'Resolution', 600);
%     exportgraphics(tl, pdf_path, 'ContentType', 'vector');
% else
%     % 旧版本 MATLAB 兼容写法
%     set(fig, 'PaperPositionMode', 'auto');
%     print(fig, png_path, '-dpng', '-r600');
%     print(fig, pdf_path, '-dpdf', '-painters');
% end
% 
% fprintf('Saved figure to:\n%s\n%s\n', png_path, pdf_path);

end

%% ============================================================
% Local functions
%% ============================================================

function A = close_theta(A)
    % Close periodic theta direction.
    % Input A: [Nr, Nt]
    % Output A: [Nr, Nt+1]
    A = [A, A(:, 1)];
end

function plot_field(X, Y, Z, cmin, cmax, ttl, fontName, axisFontSize, titleFontSize, cbFontSize, lineWidth)
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
        'FontName', fontName, ...
        'FontSize', axisFontSize, ...
        'LineWidth', lineWidth, ...
        'TickDir', 'out', ...
        'Layer', 'top', ...
        'XGrid', 'off', ...
        'YGrid', 'off', ...
        'ZGrid', 'off');

    caxis([cmin, cmax]);

    cb = colorbar;
    cb.FontSize = cbFontSize;
    cb.TickDirection = 'out';

    title(ttl, ...
        'Interpreter', 'latex', ...
        'FontSize', titleFontSize, ...
        'FontName', fontName);

    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', axisFontSize);
    ylabel('$y$', 'Interpreter', 'latex', 'FontSize', axisFontSize);
end