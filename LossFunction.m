%% 损失函数，包括 回归函数 和 分类函数
%% 多分类问题转化为高维0,1问题，如1/2/3分类问题，转化为三维(1,0,0）,(0,1,0),(0,0,1)
%% 假设多分类问题已经做好转化

function [L, dL] = LossFunction(predict, real, str1, str2, Delta)
    if nargin < 2 || class(predict) ~= "double" || class(real) ~= "double"
        fprintf("Please input predict label and real label!\n");
        return;
    elseif nargin == 2
        str1 = "Classification";
        str2 = "Cross_Entropy_Loss";
    elseif nargin == 3
        if str1 == "Regression"
            str2 = "MSE";
        elseif str1 == "Classification"
            str2 = "Cross_Entropy_Loss";
        else
            fpintf("Please input the problem to solve: Regression or Classification\n");
        end
    elseif nargin == 4
        if str2 == "Huber"
            Delta = 1.5;
        end
            elseif nargin == 4
    elseif nargin > 5
        fprintf("Input data so many parameters!");
        return;
    end
    
    L = zeros(size(real));
    dL = zeros(size(real));
    if str1 == "Regression"
        if str2 == "MSE"
            %% 平方损失函数 (predict - real)^2, 其导数 predict-real
            L = (predict - real).^2;
            dL = predict - real;
        elseif str2 == "MAE"
           %% 绝对值损失函数 |predict - real|, 其导数 +1 or -1
            temp = predict - real;
            L = abs(temp);
            dL = ones(size(temp));
            dL(temp<0) = -1;
        elseif str2 == "Huber"
           %% Huber 损失函数 0.5*(predict-real)^2 ... |predict-real|<=p, 其导数 predict-real 
           %%               p*|predict-real|-0.5*p^2 ... |predict-real|>p, 其导数 -p - 0.5*p^2 or +p - 0.5*p^2
            temp = predict - real;
            index1 = abs(temp) <= Delta;
            index2_1 = temp > Delta;
            index2_2 = temp < -Delta;
            L(index1) = 0.5*(temp(index1)).^2;
            L(index2_1) = Delta*temp(index2_1) - 0.5*Delta^2;
            L(index2_2) = -Delta*temp(index2_2) - 0.5*Delta^2;
            dL =temp;
            dL(index2_1) = Delta - 0.5*Delta^2;
            dL(index2_2) = -Delta - 0.5*Delta^2;        
        end
    elseif str1 == "Classification"
        if str2 == "Logistic_Loss"
           %% L = log(1+exp(-real*predict)), 其导数 dL = -real*exp(-real*predict)/(1+exp(-real*predict))
           L = log(1+exp(-real.*predict));
           dL = -real.*exp(-real.*predict)./(1+exp(-real.*predict));
        elseif str2 == "Cross_Entropy_Loss"
           %% 交叉熵损失函数 L=-1/m*sum{sum{log[((P_ij)^t_ij)*(1-P_ij)^(1-t_ij)]}}, i=1...C, j=1...M, C类问题M个样本（M行） 
           L = real.*(log(predict)) + (1-real).*log(1-predict);
           dL = real./predict - (1-real)./(1-predict);
        elseif str2 == "Hinge_Loss"
           %% Hinge Loss 函数(一般用于 SVM), L = max(0, 1-real*predict), 其导数 0 or -real
            temp = max(0,1 - real.*predict);
            index2 = temp > 0;
            L = temp;
            dL = temp;
            dL(index2) = -real(index2);
        elseif str2 == "exp_Loss"
            %% 指数损失函数(一般用于 AdaBoost)
            L = exp(-real.*predict);
            dL = -real.*exp(-real.*predict);
        elseif str2 == "Modified_Huber_Loss"
            %% Modified Huber Loss 函数 L = (max(0, 1-real*predict))^2 or -4*real*predict
            temp = real.*predict;
            index1 = temp >= -1;
            index2 = temp < -1;
            L(index1) = (max(0,1-real.*predict)).^2;
            L(index2) = -4*real.*predict;
            dL(index1) = 2*real.*(temp-1);
            dL(index2) = -4*real;
        end
    end
end