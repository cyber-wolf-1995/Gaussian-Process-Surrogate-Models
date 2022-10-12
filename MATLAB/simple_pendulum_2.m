%% Pendulum Parameters
clc; clear all;
m = 1;                              % Mass in kg
l = 2;                              % Length in m
dt = 0.01;                          % Time step in seconds
t0 = 0;                             % Initial Time in seconds
T = 30;                             % Terminal Time in seconds
t = t0:dt:T;                        % Time vector
N = length(t);
b = 0.1*ones(1,N);                  % Damping
b = awgn(b,10,'measured');
g = 9.81;                           % Acceleration due to gravity
u_train =  0.1*sin((2*pi/(0.1*T))*t);
u_test = 0.1*sin((2*pi/(0.1*T))*t);
%% Curate Data

theta_train = zeros(1,N);
omega_train = zeros(1,N);

for i=1:N-1
    theta_train(1,i+1) = theta_train(1,i) + omega_train(1,i)*dt;
    omega_train(1,i+1) = omega_train(1,i) + dt*(u_train(1,i) - b(1,i)*omega_train(1,i) - m*g*l*sin(theta_train(1,i)))/(m*l*l);
end

z_train = zeros(2,N-1);
for i=1:N-1
    z_train(1,i) = theta_train(1,i+1) - theta_train(1,i) - dt*omega_train(1,i);
    z_train(2,i) = omega_train(1,i+1) - omega_train(1,i) + dt*(m*g*l*sin(theta_train(1,i)) - u_train(1,i) + 0.1*omega_train(1,i))/(m*l*l);
end

theta_test = zeros(1,N);
omega_test = zeros(1,N);

for i=1:N-1
    theta_test(1,i+1) = theta_test(1,i) + omega_test(1,i)*dt;
    omega_test(1,i+1) = omega_test(1,i) + dt*(u_test(1,i) - b(1,i)*omega_test(1,i) - m*g*l*sin(theta_test(1,i)))/(m*l*l);
end

z_test = zeros(2,N-1);
for i=1:N-1
    z_test(1,i) = theta_test(1,i+1) - theta_test(1,i) - dt*omega_test(1,i);
    z_test(2,i) = omega_test(1,i+1) - omega_test(1,i) + dt*(m*g*l*sin(theta_test(1,i)) - u_test(1,i) + 0.1*omega_test(1,i))/(m*l*l);
end
theta_hat = zeros(1,N);
omega_hat = zeros(1,N);
for i=1:N-1
    theta_hat(1,i+1) = theta_hat(1,i) + omega_hat(1,i)*dt;
    omega_hat(1,i+1) = omega_hat(1,i) + dt*(u_test(1,i) - b(1,i)*omega_hat(1,i) - m*g*l*sin(theta_hat(1,i)))/(m*l*l);
end
%% Observe Response of Pendulum

plot(theta_train)
hold on
plot(omega_train)
plot(u_train)
legend('theta','omega','u')
title('Training Data')
hold off

figure;
plot(theta_test)
hold on
plot(omega_test)
plot(u_test)
legend('theta','omega','u')
title('Test Data')
hold off


%% Seperate Training and Testing Dataset
x_train = [theta_train(1:N-1)', omega_train(1:N-1)', u_train(1:N-1)'];
y1_train = z_train(1,:)';
y2_train = z_train(2,:)';
x_test = [theta_test(1:N-1)', omega_test(1:N-1)', u_test(1:N-1)'];
y1_test = z_test(1,:)';
y2_test = z_test(2,:)';
%% Train Gaussian Process
sigma0 = std(y1_train);
sigmaF0 = sigma0;
d = size(x_train,2);
sigmaM0 = 10*ones(d,1);
gpr_theta = fitrgp(x_train,y1_train,'Basis','constant','FitMethod','exact',...
'PredictMethod','exact','KernelFunction','ardsquaredexponential',...
'KernelParameters',[sigmaM0;sigmaF0],'Sigma',sigma0,'Standardize',1);


sigma0 = std(y2_train);
sigmaF0 = sigma0;
d = size(x_train,2);
sigmaM0 = 10*ones(d,1);
gpr_omega = fitrgp(x_train,y2_train,'Basis','constant','FitMethod','exact',...
'PredictMethod','exact','KernelFunction','ardsquaredexponential',...
'KernelParameters',[sigmaM0;sigmaF0],'Sigma',sigma0,'Standardize',1);


ypred1 = predict(gpr_theta, x_test);
ypred1 = ypred1' + theta_hat(1:end-1) + omega_hat(1:end-1)*dt;
theta_act = theta_test(2:end);

ypred2 = predict(gpr_omega, x_test);
ypred2 = ypred2' + omega_hat(1,i) + dt*(-m*g*l*sin(theta_hat(1:end-1)) + u_test(1:end-1) - 0.1*omega_hat(1:end-1))/(m*l*l);
omega_act = omega_test(2:end);
%% Plot Test Results
 figure;
 
 plot(theta_act)
 hold on
 plot(ypred1)
 legend('Actual','Predicted')
 title('GP Prediction of theta')
 hold off

 figure;

 plot(omega_act)
 hold on
 plot(ypred2)
 legend('Actual','Predicted')
 title('GP Prediction of omega')
 hold off

