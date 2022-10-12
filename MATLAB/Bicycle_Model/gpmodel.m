clc; clear all;
load delta.mat;
load vy.mat;
load psi.mat;
load delta_test.mat;
load vy_test.mat;
load psi_test.mat;

delta_noise = awgn(delta, 60);
vy_noise = awgn(vy,40);
psi_noise = awgn(psi, 60);

%% training Data
n = length(vy_noise);
delta_train = delta_noise(1:n-1);
vy_train = vy_noise(1:n-1);
psi_train = psi_noise(1:n-1);

y1 = vy_noise(2:n);
y2 = psi_noise(2:n);

x = [vy_train, psi_train, delta_train];

n = 4001;
xtest = [vy_test(1:n-1), psi_test(1:n-1), delta_test(1:n-1)];

gpr_vy =  fitrgp(x,y1,'KernelFunction','squaredExponential', 'FitMethod','sr','PredictMethod','sd', 'ActiveSetSize',50,'ActiveSetMethod','sgma','Standardize',true);

gpr_psi = fitrgp(x,y2,'KernelFunction','squaredExponential', 'FitMethod','sr','PredictMethod','sd', 'ActiveSetSize',50,'ActiveSetMethod','sgma','Standardize',true);

[y1pred, ~, yci1] = predict(gpr_vy, xtest);

[y2pred, ~, yci2] = predict(gpr_psi, xtest);

y1_actual = vy_test(2:n);
y2_actual = psi_test(2:n);

figure();

plot(y1pred,'b');
hold on
plot(y1_actual,'r');
plot(yci1(:,1),'k--')
plot(yci1(:,2),'k--')
hold off
title('lateral velocity')
xlabel('Time (s)')
ylabel('lateral velocity (m/s)')
legend('GP model', 'Physics Model')

figure();
plot(y2pred,'b');
hold on
plot(y2_actual,'r');
plot(yci2(:,1),'k--')
plot(yci2(:,2),'k--')
hold off
title('yaw velocity')
xlabel('Time (s)')
ylabel('yaw velocity (rad/s)')
legend('GP model', 'Physics Model')

% n_train = round(0.7*n);
% idx=randperm(n,n_train);
% 
% delta_train = delta_noise(idx);
% vy_train = vy_noise(idx);
% psi_train = psi_noise(idx);
% 
% vy_noise(idx) = [];
% vy_test = vy_noise;
% 
% delta_noise(idx) = [];
% delta_test = 