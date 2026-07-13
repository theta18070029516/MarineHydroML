clear; clc; close all;

%% ============================================================
% Load data
%% ============================================================
data_path = 'fe_reconstruction_100_varboundary.mat';
S = load(data_path);

%% ============================================================
% Select sample index
%% ============================================================
for idx = 88   % 修改这里查看不同样本，例如 idx = 10;

X = squeeze(S.x_phys_grid(idx, :, :));
Y = squeeze(S.y_phys_grid(idx, :, :));

U_true  = squeeze(S.u_true_grid(idx, :, :));
U_recon = squeeze(S.u_recon_grid(idx, :, :));
U_err   = abs(U_true - U_recon);

F_true  = squeeze(S.f_true_grid(idx, :, :));
F_recon = squeeze(S.f_recon_grid(idx, :, :));
F_err   = abs(F_true - F_recon);

k_val = S.k_values(idx);

% ============================================================
% Close periodic theta direction
% Each field is [Nr, Nt], so append the first theta column.
% ============================================================
X       = [X,       X(:,1)];
Y       = [Y,       Y(:,1)];

U_true  = [U_true,  U_true(:,1)];
U_recon = [U_recon, U_recon(:,1)];
U_err   = [U_err,   U_err(:,1)];

F_true  = [F_true,  F_true(:,1)];
F_recon = [F_recon, F_recon(:,1)];
F_err   = [F_err,   F_err(:,1)];

% Relative errors
err_u = S.err_u_each(idx);
err_f = S.err_f_each(idx);

%% ============================================================
% Figure settings: publication style
%% ============================================================
fig = figure('Color', 'w', ...
             'Units', 'centimeters', ...
             'Position', [3, 3, 34, 16]);

tl = tiledlayout(2, 3, ...
    'TileSpacing', 'compact', ...
    'Padding', 'loose');

fontName = 'Arial';
titleFontSize = 11;
labelFontSize = 10;
cbFontSize = 9;

colormap(turbo);

%% ============================================================
% Consistent color ranges
%% ============================================================
u_min = min([U_true(:); U_recon(:)]);
u_max = max([U_true(:); U_recon(:)]);

f_min = min([F_true(:); F_recon(:)]);
f_max = max([F_true(:); F_recon(:)]);

uerr_max = max(U_err(:));
ferr_max = max(F_err(:));

%% ============================================================
% Row 1: u_true, u_recon, |error|
%% ============================================================

nexttile(1);
pcolor(X, Y, U_true);
shading interp;
axis equal tight;
box on;
set(gca, 'FontName', fontName, 'FontSize', labelFontSize, ...
         'LineWidth', 0.8, 'TickDir', 'out');
caxis([u_min, u_max]);
cb = colorbar;
cb.FontSize = cbFontSize;
title(sprintf('$u_{\\mathrm{true}}$'), ...
      'Interpreter', 'latex', ...
      'FontSize', titleFontSize);

nexttile(2);
pcolor(X, Y, U_recon);
shading interp;
axis equal tight;
box on;
set(gca, 'FontName', fontName, 'FontSize', labelFontSize, ...
         'LineWidth', 0.8, 'TickDir', 'out');
caxis([u_min, u_max]);
cb = colorbar;
cb.FontSize = cbFontSize;
title(sprintf('$u_{\\mathrm{recon}}$'), ...
      'Interpreter', 'latex', ...
      'FontSize', titleFontSize);

nexttile(3);
pcolor(X, Y, U_err);
shading interp;
axis equal tight;
box on;
set(gca, 'FontName', fontName, 'FontSize', labelFontSize, ...
         'LineWidth', 0.8, 'TickDir', 'out');
caxis([0, uerr_max + eps]);
cb = colorbar;
cb.FontSize = cbFontSize;
title(sprintf('$|u_{\\mathrm{true}}-u_{\\mathrm{recon}}|$'), ...
      'Interpreter', 'latex', ...
      'FontSize', titleFontSize);

%% ============================================================
% Row 2: f_true, f_recon, |error|
%% ============================================================

nexttile(4);
pcolor(X, Y, F_true);
shading interp;
axis equal tight;
box on;
set(gca, 'FontName', fontName, 'FontSize', labelFontSize, ...
         'LineWidth', 0.8, 'TickDir', 'out');
caxis([f_min, f_max]);
cb = colorbar;
cb.FontSize = cbFontSize;
title(sprintf('$f_{\\mathrm{true}}$'), ...
      'Interpreter', 'latex', ...
      'FontSize', titleFontSize);

nexttile(5);
pcolor(X, Y, F_recon);
shading interp;
axis equal tight;
box on;
set(gca, 'FontName', fontName, 'FontSize', labelFontSize, ...
         'LineWidth', 0.8, 'TickDir', 'out');
caxis([f_min, f_max]);
cb = colorbar;
cb.FontSize = cbFontSize;
title(sprintf('$f_{\\mathrm{recon}}$'), ...
      'Interpreter', 'latex', ...
      'FontSize', titleFontSize);

nexttile(6);
pcolor(X, Y, F_err);
shading interp;
axis equal tight;
box on;
set(gca, 'FontName', fontName, 'FontSize', labelFontSize, ...
         'LineWidth', 0.8, 'TickDir', 'out');
caxis([0, ferr_max + eps]);
cb = colorbar;
cb.FontSize = cbFontSize;
title(sprintf('$|f_{\\mathrm{true}}-f_{\\mathrm{recon}}|$'), ...
      'Interpreter', 'latex', ...
      'FontSize', titleFontSize);

%% ============================================================
% Global title
%% ============================================================
sgtitle(sprintf(['FE reconstruction on variable-boundary sample %d |', ...
                 '$k = %.3f$,  $E_u = %.3e$,  $E_f = %.3e$'], ...
                 idx, k_val, err_u, err_f), ...
        'Interpreter', 'latex', ...
        'FontSize', 13, ...
        'FontWeight', 'bold');

% %% ============================================================
% % Optional: save figure
% %% ============================================================
% save_dir = './figures_fe_reconstruction';
% if ~exist(save_dir, 'dir')
%     mkdir(save_dir);
% end
% 
% exportgraphics(fig, ...
%     fullfile(save_dir, sprintf('FE_reconstruction_sample_%03d.png', idx)), ...
%     'Resolution', 600);
% 
% exportgraphics(fig, ...
%     fullfile(save_dir, sprintf('FE_reconstruction_sample_%03d.pdf', idx)), ...
%     'ContentType', 'vector');
end

function A_closed = close_periodic_theta(A)
    A_closed = [A, A(:,1)];
end