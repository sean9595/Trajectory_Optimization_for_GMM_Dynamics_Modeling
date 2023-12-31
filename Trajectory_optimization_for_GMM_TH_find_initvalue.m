%%
%Find proper init value
close all; clear all; clc;

%%
fs = 1000; % sampling frequency
Ts = 1/fs; % sampling time 
EndTime = 10;
NofMultipleSign = 20;
NofParam = 4;
means = [-1,1];
time_ = 0:Ts:EndTime-Ts;
N = length(time_);
%%
Max_rep_num = 10000;
loss_min_thres = -0.5;
loss_max_thres = 0.5;
for j = 1:Max_rep_num
% Parameter init
% `
    a = zeros(NofMultipleSign,1);
    b = zeros(NofMultipleSign,1);
    omega = zeros(NofMultipleSign,1);
    bias = zeros(NofMultipleSign,1);
    
    for i = 1:NofMultipleSign
       a(i) = rand;
       b(i) = rand;
       omega(i) = rand;
       bias(i) = rand;
    end
    delta_init = zeros(NofMultipleSign*NofParam,1);
    for i = 1:NofMultipleSign
        delta_init(1+4*(i-1)) = a(i);
        delta_init(2+4*(i-1)) = b(i);
        delta_init(3+4*(i-1)) = omega(i);
        delta_init(4+4*(i-1)) = bias(i);
    end
    % trajectory optimization
    f_c = @(delta) Test_Kmeans_cost_function(delta, N, means,Ts, NofMultipleSign);
    A = [];
    b_con = [];
    Aeq = [];
    beq = [];
    lb = [];
    ub = [];
    nonlcon = [];
    % options = optimoptions('fmincon','Display','iter','Algorithm','interior-point', 'MaxFunctionEvaluations',300000, 'ConstraintTolerance',1e-20);
    options = optimoptions('fmincon','Display','iter','Algorithm','interior-point', 'MaxFunctionEvaluations',3000000);
    delta_calc = fmincon(f_c, delta_init,A,b_con,Aeq,beq,lb,ub,nonlcon,options);
    q_opt = zeros(N, 1);
    for i=1:NofMultipleSign
        a_opt(i) = delta_calc(1+4*(i-1));
        b_opt(i) = delta_calc(2+4*(i-1));
        omega_opt(i) = delta_calc(3+4*(i-1));
        bias_opt(i) = delta_calc(4+4*(i-1));
    end
    for t = 1:N
        for i = 1:NofMultipleSign
            q_opt(t) = q_opt(t) + a_opt(i)*sin(omega_opt(i)*t*Ts)-b_opt(i)*cos(omega_opt(i)*t*Ts)+bias_opt(i);
        end
    end
    [~, Center] = kmeans(q_opt,length(means));
    Center_sorted = sort(Center);
    if ((Center_sorted(1)-(means(1))<loss_max_thres)&(Center_sorted(1)-(means(1))>loss_min_thres))
        if ((Center_sorted(2)-(means(2))<loss_max_thres)&(Center_sorted(2)-(means(2))>loss_min_thres))
            fprintf("Found init value, %d\n", i);
            break
        end
    end
end
%%
figure(1)
subplot(4,1,1)
plot(a);
grid on

subplot(4,1,2)
plot(b);
grid on

subplot(4,1,3)
plot(omega);
grid on

subplot(4,1,4)
plot(bias);
grid on
%%
figure(2)
plot(q_opt);
grid on
%%
[idx, Center] = kmeans(q_opt,length(means));