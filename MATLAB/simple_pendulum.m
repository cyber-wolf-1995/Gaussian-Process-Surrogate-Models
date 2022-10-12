clc; clear all;
%% Simulation Parameters
dt = 0.1;
N = 1000;
theta = zeros(N,1);
omega = zeros(N,1);
time = 1:N;
u = sin(time);
m = 1;
L = 1;
b = 0.6;
g =9.81;
%% Euler Forward Calculation
for t = 1:N-1
    if theta(t) > 2*pi
        theta(t) = mod(theta(t), 2*pi);
    end
    theta(t+1) = theta(t) + omega(t)*dt;
    if theta(t+1) > 2*pi
        theta(t+1) = mod(theta(t+1), 2*pi);
    end
    omega(t+1) = u(t)*dt - (g*dt/L)*sin(theta(t)) + (1 - (b*dt/(m*L*L)))*omega(t);
end
%% Plot
plot(theta);
hold on
plot(omega)
%% Residuals
et = theta(2:end) - theta(1:end-1) - dt*omega(1:end-1);
ew = omega(2:end) + (g*dt/L)*sin(theta(t)) - (1 - (b*dt/(m*L*L)))*omega(t);
%% GP Modeling of Residual
gp_theta = fitrgp(u(1:end-1)',et,'KernelFunction','matern32');
gp_omega = fitrgp(u(1:end-1)',ew,'KernelFunction','matern32');
[etpred,~,etint] = predict(gp_theta,u(1:end-1)');
[ewpred,~,ewint] = predict(gp_omega,u(1:end-1)');
figure;
plot(et)
hold on
plot(etpred)

figure;
plot(ew)
hold on
plot(ewpred)


