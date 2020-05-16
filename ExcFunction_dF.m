%% �����������䵼��
function [Y, dY]  = ExcFunction_dF(X, ExcFunc)
    %% �����������
    if nargin == 0 || class(X) ~= "double"
        fprintf("Please input data for excitation function!");
        return;
    elseif nargin == 1
        ExcFunc = "sigmoid";
    elseif nargin > 2
        fprintf("Input data so many parameters!");
        return;
    end
    if ExcFunc == "sigmoid"
        %% sigmoid ���� 1/(1+exp(net))
        Y = 1 ./ (1 + exp(-X));
        dY = Y .* (1 - Y);
    elseif ExcFunc == "tanh"
        %% tanh ���� (exp(net) - exp(-net))./(exp(net) + exp(-net))
        Y = (exp(X) - exp(-X))./(exp(X) + exp(-X));
        dY = 1 - Y.^2;
    elseif ExcFunc == "ReLU"
        %% ReLU ����
        Y = max(0,X);
        dY = Y;
        dY(X > 0) = 1;
    elseif ExcFunc == "PReLU"
        %% PReLU ����
        alpha = 0.01;
        Y = max(alpha*X,X);
        dY = alpha * ones(size(Y));
        dY(X > 0) = 1;
    elseif ExcFunc == "ELU"
        %% ELU ����
        alpha = 0.01;
        Y = X;
        Y(X <= 0) = alpha * (exp(X(X <= 0)) - 1);
        dY = ones(size(Y));
        dY(X <= 0) = alpha * (exp(X(X <= 0)));
    else
        fprintf("\nThere is no excitation function named %s!\n",ExcFunc);
    end
end