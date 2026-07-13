clear; clc; close all;

%% ============================================================
% Load data
%% ============================================================
data_path = 'circle_f0_flux_cos_theta_transformer_fe_k_0p5_1p0_1p5.mat';
S = load(data_path);

%% ============================================================
% Basic data
%% ============================================================
X_base = S.x_grid;          % [Nr, Nt]
Y_base = S.y_grid;          % [Nr, Nt]

k_values = S.k_values(:);   % [3, 1]

num_cases = numel(k_values);

%% ============================================================
% Figure style
%% ============================================================
fontName = 'Arial';
axisFontSize = 10;
titleFontSize = 12;
sgTitleFontSize = 14;
cbFontSize = 9;
lineWidth = 0.85;

%% ============================================================
% Loop over k cases
%% ============================================================
for icase = 1:num_cases

    k_val = k_values(icase);

    %% ------------------------------------------------------------
    % Read fields
    %% ------------------------------------------------------------
    U_pred   = squeeze(S.u_pred_grid(icase, :, :));      % [Nr, Nt]
    U_theory = squeeze(S.u_theory_grid(icase, :, :));    % [Nr, Nt]
    U_abs_err = abs(U_pred - U_theory);

    %% ------------------------------------------------------------
    % Relative error
    %% ------------------------------------------------------------
    if isfield(S, 'err_u_pred_vs_theory')
        rel_err = S.err_u_pred_vs_theory(icase);
    else
        rel_err = norm(U_pred(:) - U_theory(:), 2) / ...
                  (norm(U_theory(:), 2) + eps);
    end

    %% ------------------------------------------------------------
    % Close periodic theta direction
    %% ------------------------------------------------------------
    X = close_theta(X_base);
    Y = close_theta(Y_base);

    U_pred_closed    = close_theta(U_pred);
    U_theory_closed  = close_theta(U_theory);
    U_abs_err_closed = close_theta(U_abs_err);

    %% ------------------------------------------------------------
    % Inner and outer boundary curves
    %% ------------------------------------------------------------
    r_inner = S.r_inner;
    r_outer = S.r_outer;

    theta_b = linspace(0, 2*pi, 512);

    xb_inner = r_inner * cos(theta_b);
    yb_inner = r_inner * sin(theta_b);

    xb_outer = r_outer * cos(theta_b);
    yb_outer = r_outer * sin(theta_b);

    %% ------------------------------------------------------------
    % Color limits
    % Predicted and theoretical fields share same color range.
    %% ------------------------------------------------------------
    u_min = min([U_pred_closed(:); U_theory_closed(:)]);
    u_max = max([U_pred_closed(:); U_theory_closed(:)]);

    err_max = max(U_abs_err_closed(:));
    if err_max <= 0
        err_max = eps;
    end

    %% ------------------------------------------------------------
    % Create figure
    %% ------------------------------------------------------------
    fig = figure('Color', 'w', ...
                 'Units', 'centimeters', ...
                 'Position', [3, 3, 28, 8.5]);

    tl = tiledlayout(1, 3, ...
        'TileSpacing', 'compact', ...
        'Padding', 'loose');

    colormap(turbo);

    %% ------------------------------------------------------------
    % Subplot 1: prediction
    %% ------------------------------------------------------------
    nexttile(1);
    plot_field(X, Y, U_pred_closed, u_min, u_max, ...
        '$u_{\mathrm{pred}}$', ...
        fontName, axisFontSize, titleFontSize, cbFontSize, lineWidth);

%     hold on;
%     plot(xb_inner, yb_inner, 'k-', 'LineWidth', 1.0);
%     plot(xb_outer, yb_outer, 'k-', 'LineWidth', 1.0);
%     hold off;

    %% ------------------------------------------------------------
    % Subplot 2: theory
    %% ------------------------------------------------------------
    nexttile(2);
    plot_field(X, Y, U_theory_closed, u_min, u_max, ...
        '$u_{\mathrm{theory}}$', ...
        fontName, axisFontSize, titleFontSize, cbFontSize, lineWidth);

%     hold on;
%     plot(xb_inner, yb_inner, 'k-', 'LineWidth', 1.0);
%     plot(xb_outer, yb_outer, 'k-', 'LineWidth', 1.0);
%     hold off;

    %% ------------------------------------------------------------
    % Subplot 3: absolute error
    %% ------------------------------------------------------------
    nexttile(3);
    plot_field(X, Y, U_abs_err_closed, 0, err_max, ...
        '$|u_{\mathrm{pred}}-u_{\mathrm{theory}}|$', ...
        fontName, axisFontSize, titleFontSize, cbFontSize, lineWidth);

%     hold on;
%     plot(xb_inner, yb_inner, 'k-', 'LineWidth', 1.0);
%     plot(xb_outer, yb_outer, 'k-', 'LineWidth', 1.0);
%     hold off;

    %% ------------------------------------------------------------
    % Global title
    %% ------------------------------------------------------------
    sgtitle(sprintf(['Standard circular annulus, $f=0$, $g(\\theta)=\\cos\\theta$  |  ', ...
                     '$k=%.2f$,  $E_u=%.3e$'], ...
                     k_val, rel_err), ...
            'Interpreter', 'latex', ...
            'FontName', fontName, ...
            'FontSize', sgTitleFontSize, ...
            'FontWeight', 'bold');

end

%% ============================================================
% Local functions
%% ============================================================

function A = close_theta(A)
    % Close periodic theta direction.
    % Input : [Nr, Nt]
    % Output: [Nr, Nt+1]
    A = [A, A(:, 1)];
end

function plot_field(X, Y, Z, cmin, cmax, ttl, ...
                    fontName, axisFontSize, titleFontSize, cbFontSize, lineWidth)

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
        'FontName', fontName, ...
        'FontSize', titleFontSize);

    xlabel('$x$', 'Interpreter', 'latex', 'FontSize', axisFontSize);
    ylabel('$y$', 'Interpreter', 'latex', 'FontSize', axisFontSize);
end