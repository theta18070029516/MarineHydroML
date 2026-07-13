clear; clc; close all;

%% =======================
%  Load data
% ========================
data = load('sno_100_flux_samples.mat');

pod_coords = data.pod_coords;          % [Npod, 2]
u_true_pod = data.u_true_pod;          % [Nsamples, Npod]
u_pred_pod = data.u_pred_pod;          % [Nsamples, Npod]
err_sno_pod = data.err_sno_pod;        % [Nsamples, 1]

Nsamples = 20; %size(u_true_pod, 1);

x = pod_coords(:, 1);
y = pod_coords(:, 2);

%% =======================
%  Output folder
% ========================
% fig_dir = 'sno_pod_figures';
% if ~exist(fig_dir, 'dir')
%     mkdir(fig_dir);
% end

%% =======================
%  Global plot settings
% ========================
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 12);
set(groot, 'defaultTextFontSize', 12);
set(groot, 'defaultAxesLineWidth', 1.0);
set(groot, 'defaultFigureColor', 'w');

%% =======================
%  Plot all samples
% ========================
for idx = 1:3 %1:Nsamples

    u_true = u_true_pod(idx, :)';
    u_pred = u_pred_pod(idx, :)';
    abs_err = abs(u_pred - u_true);

    % 保证 true / pred 使用相同色标，便于直接对比
    cmin = min([u_true; u_pred]);
    cmax = max([u_true; u_pred]);

    % 误差色标单独设置
    emin = 0;
    emax = max(abs_err);

    fig = figure('Units', 'centimeters', ...
                 'Position', [3, 3, 25, 7.2], ...
                 'Color', 'w');

    tiledlayout(1, 3, ...
        'Padding', 'compact', ...
        'TileSpacing', 'compact');

    %% ---- Subplot 1: true u ----
    nexttile;
    scatter(x, y, 16, u_true, 'filled');
    axis equal tight;
    box on;
    colormap(gca, turbo);
    caxis([cmin, cmax]);
    cb = colorbar;
%     cb.Label.String = '$u$';
    cb.Label.Interpreter = 'latex';
    title('$P_{\mathrm{true}}$', 'Interpreter', 'latex');
    xlabel('$x$', 'Interpreter', 'latex');
    ylabel('$y$', 'Interpreter', 'latex');
    set(gca, 'TickDir', 'out', 'Layer', 'top');

    %% ---- Subplot 2: predicted u ----
    nexttile;
    scatter(x, y, 16, u_pred, 'filled');
    axis equal tight;
    box on;
    colormap(gca, turbo);
    caxis([cmin, cmax]);
    cb = colorbar;
%     cb.Label.String = '$u$';
    cb.Label.Interpreter = 'latex';
    title('$P_{\mathrm{pred}}$', 'Interpreter', 'latex');
    xlabel('$x$', 'Interpreter', 'latex');
    ylabel('$y$', 'Interpreter', 'latex');
    set(gca, 'TickDir', 'out', 'Layer', 'top');

    %% ---- Subplot 3: absolute error ----
    nexttile;
    scatter(x, y, 16, abs_err, 'filled');
    axis equal tight;
    box on;
    colormap(gca, hot);
    caxis([emin, max(emax, 1e-12)]);
    cb = colorbar;
%     cb.Label.String = '$|u_{\mathrm{pred}}-u_{\mathrm{true}}|$';
    cb.Label.Interpreter = 'latex';
    title('$|P_{\mathrm{pred}}-P_{\mathrm{true}}|$', 'Interpreter', 'latex');
    xlabel('$x$', 'Interpreter', 'latex');
    ylabel('$y$', 'Interpreter', 'latex');
    set(gca, 'TickDir', 'out', 'Layer', 'top');

    %% ---- Global title with relative error ----
    sgtitle(sprintf('Sample %03d: Relative $L^2$ error = %.4e', ...
        idx, err_sno_pod(idx)), ...
        'Interpreter', 'latex', ...
        'FontSize', 14, ...
        'FontWeight', 'normal');

    %% ---- Save figure ----
%     save_name_png = fullfile(fig_dir, sprintf('sample_%03d_pod.png', idx));
%     save_name_pdf = fullfile(fig_dir, sprintf('sample_%03d_pod.pdf', idx));
% 
%     exportgraphics(fig, save_name_png, 'Resolution', 300);
%     exportgraphics(fig, save_name_pdf, 'ContentType', 'vector');
% 
%     close(fig);
end

% fprintf('Finished plotting %d samples.\nFigures saved in folder: %s\n', Nsamples, fig_dir);