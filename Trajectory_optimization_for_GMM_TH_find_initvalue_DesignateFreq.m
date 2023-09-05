%%
%Find proper init value
close all; clear all; clc;

%%
fs = 1000; % sampling frequency
Ts = 1/fs; % sampling time 
EndTime = 10;
NofMultipleSign = 5;
NofParam = 2;
% means = [-1.5,1.5];
means = [-2,-1,0,1,2];
time_ = 0:Ts:EndTime-Ts;
N = length(time_);
%% Designate frequency
% Min_Freq = 10;
Max_Freq = 10;
FreqList = [10,20,30,40,50];
%%
Max_rep_num = 10000;
loss_min_thres = -0.1;
loss_max_thres = 0.1;
for j = 1:Max_rep_num
% Parameter init
% `
    a = zeros(NofMultipleSign,1);
    b = zeros(NofMultipleSign,1);
    omega = zeros(NofMultipleSign,1);
%     bias = zeros(NofMultipleSign,1);
    
    for i = 1:NofMultipleSign
       a(i) = rand;
       b(i) = rand;
       omega(i) = FreqList(i);
%        omega(i) = Max_Freq*rand;
    end
    bias = rand; 

    delta_init = zeros(NofMultipleSign*NofParam+1,1);
    for i = 1:NofMultipleSign
        delta_init(1+NofParam*(i-1)) = a(i);
        delta_init(2+NofParam*(i-1)) = b(i);
    end
    delta_init(NofParam*NofMultipleSign+1) = bias;
    
%     delta_init_GPU = gpuArray(delta_init);

    % trajectory optimization
    f_c = @(delta) Test_Kmeans_cost_function_Reduced_Param_DesignateFreq(delta, N, means,Ts, NofMultipleSign,omega);
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
    
%     delta_calc = gather(delta_calc_GPU);

    q_opt = zeros(N, 1);
    for i=1:NofMultipleSign
        a_opt(i) = delta_calc(1+NofParam*(i-1));
        b_opt(i) = delta_calc(2+NofParam*(i-1));
%         omega_opt(i) = delta_calc(3+NofParam*(i-1));
    end
    bias_opt = delta_calc(NofParam*NofMultipleSign+1);

    for t = 1:N
        for i = 1:NofMultipleSign
            q_opt(t) = q_opt(t) + a_opt(i)*sin(omega(i)*t*Ts)-b_opt(i)*cos(omega(i)*t*Ts);
        end
    end
    q_opt = q_opt + bias_opt;

    [~, Center] = kmeans(q_opt,length(means));
    Center_sorted = sort(Center);
    Count = 0;
    for i=1:length(means)
        if (Center_sorted(i)-(means(i))<loss_max_thres)&(Center_sorted(i)-(means(i))>loss_min_thres)
            Count = Count + 1;
        end
    end
    if Count==length(means)
        fprintf("Found init value, %d\n", i);
        break
    end
%     if ((Center_sorted(1)-(means(1))<loss_max_thres)&(Center_sorted(1)-(means(1))>loss_min_thres))
%         if ((Center_sorted(2)-(means(2))<loss_max_thres)&(Center_sorted(2)-(means(2))>loss_min_thres))
%             if ((Center_sorted(3)-(means(3))<loss_max_thres)&(Center_sorted(3)-(means(3))>loss_min_thres))
%                 fprintf("Found init value, %d\n", i);
%                 break
%             end
%         end
%     end
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