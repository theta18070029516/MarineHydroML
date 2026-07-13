%%% 优化后的 u 和 f 分布查看程序
clear; clc; close all;

load('sample_u_f.mat');

% --- 全局图形设置 ---
fig_width = 1000;  % 窗口宽度
fig_height = 420;  % 窗口高度
pt_size = 10;      % 散点大小（可根据网格稠密度微调）

for id = 1:10
    % 1. 创建图形并固定尺寸，设置纯白背景（避免导出时带灰色边框）
    fig = figure(id);
    set(fig, 'Position', [100, 100, fig_width, fig_height], 'Color', 'w');
    
    % 2. 左侧子图：u field
    ax1 = subplot(1,2,1);
    scatter(coords(:,1), coords(:,2), pt_size, u(id,:)/6.36, 'filled');
    % 使用 'turbo' 或 'jet' 色图更能凸显流场的高低频特征
    colormap(ax1, 'turbo'); 
    cb1 = colorbar;
    % 设置标题和坐标轴字体，增强学术规范感
    title(['u field (Sample ', num2str(id), ')'], 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('x', 'FontSize', 12); 
    ylabel('y', 'FontSize', 12);
    axis equal tight;
    set(gca, 'FontSize', 11, 'LineWidth', 1); % 加粗坐标轴边框
    
    % 3. 右侧子图：f field
    ax2 = subplot(1,2,2);
    scatter(coords(:,1), coords(:,2), pt_size, f(id,:)/164.46, 'filled');
    colormap(ax2, 'turbo');
    cb2 = colorbar;
    title(['f field (Sample ', num2str(id), ')'], 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('x', 'FontSize', 12); 
    ylabel('y', 'FontSize', 12);
    axis equal tight;
    set(gca, 'FontSize', 11, 'LineWidth', 1);
    
    % 4. 交互式查看逻辑：避免一次性弹出大量窗口
    % 如果你想一张一张地审查样本，请取消下面两行的注释：
%     disp(['Displaying Sample ', num2str(id), ' / 10. Press any key for next...']);
%     pause; 
end