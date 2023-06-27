function cost = Test_Kmeans_cost_function(delta, N, means, Ts, NofMultipleSign)
    %Param designate
    for i=1:NofMultipleSign
        a_cost(i) = delta(1+4*(i-1));
        b_cost(i) = delta(2+4*(i-1));
        omega_cost(i) = delta(3+4*(i-1));
        bias_cost(i) = delta(4+4*(i-1));
    end

    q_0 = zeros(N,1);

    for t = 1:N
        for i = 1:NofMultipleSign
            q_0(t) = q_0(t) + a_cost(i)*sin(omega_cost(i)*t*Ts)-b_cost(i)*cos(omega_cost(i)*t*Ts)+bias_cost(i);
        end
    end
    %Cost func 01
%     q_0_mean = mean(q_1);
%     cost = 0;
%     for i = 1:length(means)
%         cost = cost + norm(q_0_mean-means(i), "fro")^2;   
%     end

    %Cost func 02
%     [idx, Center] = kmeans(q_0,length(means));
%     q_mean_kmeans = zeros(length(means),1);
%     cost = 0;
%     for i = 1:length(means)
%         q_mean_kmeans(i) = Center(i);
%         cost = cost + norm(q_mean_kmeans(i)-means(i), "fro")^2;
%     end

    %Cost func 03
%     cost = 0;
%     cost_temp = 0;
%     for i = 1:length(means)
%         for j = 1:N
%             cost_temp = cost_temp + norm(q_0(j)-means(i))^2;   
%         end
%         cost = cost + cost_temp;
%     end

    %Cost func 04
    for j = 1:2:N
        cost_temp_1 = norm(q_0(j)-means(1))^2;   
        cost_temp_2 = norm(q_0(j+1)-means(2))^2;  
    end
    cost = cost_temp_1 + cost_temp_2;

end    