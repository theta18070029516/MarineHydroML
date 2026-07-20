function [a, a_theta, a_theta2] = eval_geometry_bnn(theta, geom, cfg)
%EVAL_GEOMETRY_BNN Evaluate the Python-exported periodic geometry BNN.
%
% theta may be any shape. The returned arrays preserve that shape.

    original_size = size(theta);
    theta_col = double(theta(:));
    w1 = double(geom.w1);       % [2, H]
    b1 = double(geom.b1(:).');  % [1, H]
    w2 = double(geom.w2(:));    % [H, 1]
    hidden = size(w1, 2);

    s = sin(theta_col);
    c = cos(theta_col);
    z = s * w1(1, :) + c * w1(2, :) + b1;
    z_theta = c * w1(1, :) - s * w1(2, :);
    z_theta2 = -s * w1(1, :) - c * w1(2, :);
    phase = z - pi / 4.0;

    feature = sqrt(2.0) * cos(phase);
    feature_theta = -sqrt(2.0) * sin(phase) .* z_theta;
    feature_theta2 = -sqrt(2.0) * (...
        cos(phase) .* z_theta.^2 + sin(phase) .* z_theta2);

    raw = feature * w2 / sqrt(hidden);
    raw_theta = feature_theta * w2 / sqrt(hidden);
    raw_theta2 = feature_theta2 * w2 / sqrt(hidden);

    scale = double(cfg.geom_tanh_scale);
    tanh_raw = tanh(scale * raw);
    sech2 = 1.0 - tanh_raw.^2;
    a_col = double(cfg.geom_base) + double(cfg.geom_amp) * tanh_raw;
    a_theta_col = double(cfg.geom_amp) * scale .* sech2 .* raw_theta;
    a_theta2_col = double(cfg.geom_amp) * scale .* (...
        sech2 .* raw_theta2 ...
        - 2.0 * scale .* tanh_raw .* sech2 .* raw_theta.^2);

    a = reshape(a_col, original_size);
    a_theta = reshape(a_theta_col, original_size);
    a_theta2 = reshape(a_theta2_col, original_size);
end
