clear;clc;close all;

load('fe_reconstruction_check.mat') %sno_100_test_samples.mat %fe_reconstruction_check.mat

for i = 1:5

figure('Color','w','Position',[100 100 1200 700])

subplot(2,3,1)
scatter(grid(:,1), grid(:,2), 12, u_ref(i,:), 'filled')
axis equal tight
colorbar
title('u reference')

subplot(2,3,2)
scatter(grid(:,1), grid(:,2), 12, u_recon(i,:), 'filled')
axis equal tight
colorbar
title('u reconstruction')

subplot(2,3,3)
scatter(grid(:,1), grid(:,2), 12, u_error(i,:), 'filled')
axis equal tight
colorbar
title('u error')

subplot(2,3,4)
scatter(grid(:,1), grid(:,2), 12, f_ref(i,:), 'filled')
axis equal tight
colorbar
title('f reference')

subplot(2,3,5)
scatter(grid(:,1), grid(:,2), 12, f_recon(i,:), 'filled')
axis equal tight
colorbar
title('f reconstruction')

subplot(2,3,6)
scatter(grid(:,1), grid(:,2), 12, f_error(i,:), 'filled')
axis equal tight
colorbar
title('f error')

sgtitle(sprintf('Sanple %d: RL2_u = %.3e, RL2_f = %.3e', i, rl2_u_each(i), rl2_f_each(i)))
end