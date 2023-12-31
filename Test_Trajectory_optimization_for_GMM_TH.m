%%
%FMINCON Test
%Thanks to TJK
close all; clear all; clc;

%% 
% delta = [a1]
fs = 1000; % sampling frequency
Ts = 1/fs; % sampling time 
EndTime = 10;
NofMultipleSign = 10;
NofParam = 4;

% Parameter init
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
%%
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
% NofMean = 2;
% means = [-1,1,-1,1,-1];
means = [-2,10];

delta_init = zeros(NofMultipleSign*NofParam,1);
for i = 1:NofMultipleSign
    delta_init(1+4*(i-1)) = a(i);
    delta_init(2+4*(i-1)) = b(i);
    delta_init(3+4*(i-1)) = omega(i);
    delta_init(4+4*(i-1)) = bias(i);
end
%%
time_ = 0:Ts:EndTime-Ts;
N = length(time_);
%% trajectory optimization
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
% %%
% % q_opt = zeros(N,1);
% %     for t = 1:N
% %         q_opt(t) = (delta_calc(1))*sin(delta_calc(3)*t*Ts)-(delta_calc(2))*cos(delta_calc(3)*t*Ts)+(delta_calc(4))*sin(delta_calc(6)*t*Ts)-(delta_calc(5))*cos(delta_calc(6)*t*Ts)+(delta_calc(7))*sin(delta_calc(9)*t*Ts)-(delta_calc(8))*cos(delta_calc(9)*t*Ts)+(delta_calc(10))*sin(delta_calc(12)*t*Ts)-(delta_calc(11))*cos(delta_calc(12)*t*Ts)+delta_calc(13)+delta_calc(14);
% %     end
% q_opt = zeros(N, 1);
% for t = 1:N
%     q_opt(t) = (delta_calc(1))*sin(delta_calc(3)*t*Ts)-(delta_calc(2))*cos(delta_calc(3)*t*Ts)+(delta_calc(4))*sin(delta_calc(6)*t*Ts)-(delta_calc(5))*cos(delta_calc(6)*t*Ts)+delta_calc(13);
%     q_opt(N+t) = (delta_calc(7))*sin(delta_calc(9)*t*Ts)-(delta_calc(8))*cos(delta_calc(9)*t*Ts)+(delta_calc(10))*sin(delta_calc(12)*t*Ts)-(delta_calc(11))*cos(delta_calc(12)*t*Ts)+delta_calc(14);
% end
%%
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
%%
figure(2)
plot(q_opt);
grid on
%%
[idx, Center] = kmeans(q_opt,length(means));
% %% Iteration 01
% % Take Obtained parameter into initial value for next iteration
% for i = 1:NofMultipleSign
%     delta_init(1+4*(i-1)) = delta_calc(1+4*(i-1));
%     delta_init(2+4*(i-1)) = delta_calc(2+4*(i-1));
%     delta_init(3+4*(i-1)) = delta_calc(3+4*(i-1));
%     delta_init(4+4*(i-1)) = delta_calc(4+4*(i-1));
% end
% %% trajectory optimization
% f_c = @(delta) Test_Kmeans_cost_function(delta, N, means,Ts, NofMultipleSign);
% A = [];
% b = [];
% Aeq = [];
% beq = [];
% lb = [];
% ub = [];
% nonlcon = [];
% % options = optimoptions('fmincon','Display','iter','Algorithm','interior-point', 'MaxFunctionEvaluations',300000, 'ConstraintTolerance',1e-20);
% options = optimoptions('fmincon','Display','iter','Algorithm','interior-point', 'MaxFunctionEvaluations',3000000);
% delta_calc = fmincon(f_c, delta_init,A,b,Aeq,beq,lb,ub,nonlcon,options);
% %%
% q_opt_iter02 = zeros(N, 1);
% for i=1:NofMultipleSign
%     a_opt_02(i) = delta_calc(1+4*(i-1));
%     b_opt_02(i) = delta_calc(2+4*(i-1));
%     omega_opt_02(i) = delta_calc(3+4*(i-1));
%     bias_opt_02(i) = delta_calc(4+4*(i-1));
% end
% for t = 1:N
%     for i = 1:NofMultipleSign
%         q_opt_iter02(t) = q_opt_iter02(t) + a_opt_02(i)*sin(omega_opt_02(i)*t*Ts)-b_opt_02(i)*cos(omega_opt_02(i)*t*Ts)+bias_opt_02(i);
%     end
% end
% %%
% figure(2)
% plot(q_opt_iter02);
% grid on
% %%
% [idx, Center] = kmeans(q_opt_iter02,length(means));
%% 
% % delta = [a1]
% fs = 1000; % sampling frequency
% Ts = 1/fs; % sampling time
% EndTime = 10;
% NofMultipleSign = 5;
% NofParam = 4;
% 
% % Parameter init
% a_1 = zeros(NofMultipleSign,1);
% b_1 = zeros(NofMultipleSign,1);
% omega_1 = zeros(NofMultipleSign,1);
% bias_1 = zeros(NofMultipleSign,1);
% 
% for i = 1:NofMultipleSign
%    a_1(i) = 0.1;
%    b_1(i) = 0.1;
%    omega_1(i) = 0.1;
%    bias_1(i) = 0.001;
% end
% %%
% means_1 = [30];
% 
% % delta_0 = [a_0,b_0,omega_0,a_1,b_1,omega_1,a_2,b_2,omega_2];
% delta_init_1 = zeros(NofMultipleSign*NofParam,1);
% for i = 1:NofMultipleSign
%     delta_init_1(1+4*(i-1)) = a_1(i);
%     delta_init_1(2+4*(i-1)) = b_1(i);
%     delta_init_1(3+4*(i-1)) = omega_1(i);
%     delta_init_1(4+4*(i-1)) = bias_1(i);
% end
% %%
% time_ = 0:Ts:EndTime-Ts;
% N = length(time_);
% %% trajectory optimization
% f_c_1 = @(delta_1) Test_Kmeans_cost_function(delta_1, N, means_1,Ts, NofMultipleSign);
% A = [];
% b = [];
% Aeq = [];
% beq = [];
% lb = [];
% ub = [];
% nonlcon = [];
% % options = optimoptions('fmincon','Display','iter','Algorithm','interior-point', 'MaxFunctionEvaluations',300000, 'ConstraintTolerance',1e-20);
% options = optimoptions('fmincon','Display','iter','Algorithm','interior-point', 'MaxFunctionEvaluations',300000);
% delta_calc_1 = fmincon(f_c_1, delta_init_1,A,b,Aeq,beq,lb,ub,nonlcon,options);
% %%
% q_opt_1 = zeros(N, 1);
% for i=1:NofMultipleSign
%     a_opt_1(i) = delta_calc_1(1+4*(i-1));
%     b_opt_1(i) = delta_calc_1(2+4*(i-1));
%     omega_opt_1(i) = delta_calc_1(3+4*(i-1));
%     bias_opt_1(i) = delta_calc_1(4+4*(i-1));
% end
% for t = 1:N
%     for i = 1:NofMultipleSign
%         q_opt_1(t) = q_opt_1(t) + a_opt_1(i)*sin(omega_opt_1(i)*t*Ts)-b_opt_1(i)*cos(omega_opt_1(i)*t*Ts)+bias_opt_1(i);
%     end
% end
% %%
% plot(q_opt_1);
% %%
% [idx_1, Center_1] = kmeans(q_opt_1,1);
% %%
% q_opt_tot = q_opt+q_opt_1;
% plot(q_opt_tot);
% [idx_tot, Center_tot] = kmeans(q_opt_tot,2);