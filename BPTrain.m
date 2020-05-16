%% BP 神经网络训练器
%% 多分类问题转化为高维0,1问题，如1/2/3分类问题，转化为三维(1,0,0）,(0,1,0),(0,0,1)
%% 这里假定label都已经转化好了
%% 损失函数一般用交叉熵
function net = BPTrain(Data,Label,hid,Train_rate,ExcFunc,LossFunc)
% ExcFunc: "sigmoid" or "tanh" or "ReLU" or "PReLU" or "ELU"
% LossFunction: ["Regression" "MSE"] or ["Regression" "MAE"] or ["Regression" "Huber"]
%            or ["Classification" "Logistic_Loss"] or ["Classification" "Cross_Entropy_Loss"]
%            or ["Classification" "Hinge_Loss"] or ["Classification" "exp_Loss"] 
%            or ["Classification" "Modified_Huber_Loss"]
    %% 参数检查
    ExcF = ["sigmoid" "tanh" "ReLU" "PReLU" "ELU"];
    LossF_R = ["MSE" "MAE" "Huber"];
    LossF_C = ["Logistic_Loss" "Cross_Entropy_Loss" "Hinge_Loss" "exp_Loss" "Modified_Huber_Loss"];
    if nargin < 2 
        fprintf("Please input data and label for train!");
        return;
    elseif nargin == 2
        hid = 3;
        Train_rate = 0.1;
        ExcFunc = "sigmoid";
        LossFunc = ["Regression" "MSE"];
    elseif nargin == 2
        Train_rate = 0.1;
        ExcFunc = "sigmoid";
        LossFunc = ["Regression" "MSE"];
    elseif nargin == 3
        ExcFunc = "sigmoid";
        LossFunc = ["Regression" "MSE"];
    elseif nargin == 4
        LossFunc = ["Regression" "MSE"];
    end
    if length(Data(:,1))~=length(Label(:,1))
        fprintf("Please enter a label that matches the data!\n");
        return;
    end
    if int32(hid) ~= hid
        fprintf("Please input right hidden layer for train!");
        fprintf("eg: [3 4] means two hidden layers, the first layer has 3 nodes, the second layer has 4 nodes.\n");
    end
    if hid <= 0
        fprintf("Please input right hidden layer for train!");
        fprintf("eg: [3 4] means two hidden layers, the first layer has 3 nodes, the second layer has 4 nodes.\n");
    end
    if Train_rate <=0
        fprintf("Please input right Training rate(>0) length for iteration W and others!\n");
    end
    if ~ismember(ExcFunc,ExcF)
        fprintf("Please input right excitation function name for train!\n");
        fprintf('eg: "sigmoid" or "tanh" or "ReLU" or "PReLU" or "ELU"\n');
    end
    if length(LossFunc) ~= 2 || ~((LossFunc(1)=="Regression"&&ismember(LossFunc(2),LossF_R))||(LossFunc(1)=="Classification"&&ismember(LossFunc(2),LossF_C)))
        fprintf("Please input right name of loss function");
        fprintf('eg: ["Regression" "MSE"] or ["Regression" "MAE"] or ["Regression" "Huber"]');
        fprintf('or ["Classification" "Logistic_Loss"] or ["Classification" "Cross_Entropy_Loss"]');
        fprintf('or ["Classification" "Hinge_Loss"] or ["Classification" "exp_Loss"]');
        fprintf('or ["Classification" "Modified_Huber_Loss"]');
    end
    %% 参数检查结束
    
    %% 定义基本迭代参数
    temp1 = length(unique(Label,'rows'));
    HID = [hid temp1]; %% 输入层 + 隐藏层 + 输出层 - 1


    W = cell(length(HID),2);  %% 权重相关, W, dW， 输入层 + 隐藏层 + 输出层 - 1
    H = cell(1 + length(HID),3);  %% 激励函数相关, H_in=XW-B, H_out=f(H_in), f'  输入层 + 隐藏层 + 输出层
    Bias = cell(length(HID),2); %% 偏移量相关, Bias, dBias   输入层 + 隐藏层 + 输出层 - 1
    L = cell(1,2);           %% 损失函数相关, L, dL  输出层
    L_annealing = cell(1,2);           %% 损失函数相关, L, dL  输出层
    e = cell(length(HID),1);  %%  输入层 + 隐藏层 + 输出层 - 1

    Layer = [length(Data(1,:)) HID];

    H{1,1} = zeros(size(Data(1,:)));
    H{1,2} = zeros(size(Data(1,:)));
    for i = 1:length(HID)
        H{i+1,1} = zeros(1,HID(i));
        W{i,1} = 2*rand(length(H{i,1}),length(H{i+1,1})) - 1;
        Bias{i,1} = 2*rand(1,HID(i)) - 1;
    end
    %% 退火温度 初始化
    Tmax = 1000;
    Tmin = 0.01;
    T = Tmax;
    W_annealing = W;
    Bias_annealing = Bias;
    H_annealing = H;
    
    NLoop = 0;  %%% 循环迭代次数
    Delta = 0.00001;  %%% 总损失误差变化 dL
    L_err = 100;
% % %     while (T > Tmin) && (NLoop < 10000)
    LPlot = [];
    while T > Tmin
%     while NLoop < 10000
        %%% 退火温度小于最低退火温度 或 总损失误差变化小于Delta 或 循环大于最大循环次数NLoop 决定循环停止
% % %         if L_err < Delta
% % %             break;
% % %         end
        NLoop = NLoop + 1;
        T = 0.99*T; %% 
        L_err = 0;
        for i_event = 1:length(Data(:,1))
            %% 以事例为单位循环
            Data0 = Data(i_event,:);
            real_Label0 = Label(i_event,:);

            H{1,1} = Data0;
            H{1,2} = Data0;
            for j = 1:length(HID)
                %% 根据权重计算各节点结果
                W_data = W{j,1};
                H_data = H{j,2};
                H{j+1,1} = H_data*W_data - Bias{j};
                [H{j+1,2}, H{j+1,3}] = ExcFunction_dF(H{j+1,1},ExcFunc);
            end
            %% 单一事例损失函数误差
            [L{1}, L{2}] = LossFunction(H{end,2}, real_Label0, LossFunc(1), LossFunc(2), 1.5);
            L_err = L_err + sum(abs(L{2}));
            LPlot = [LPlot sum(L{1})];
            %% 根据损失函数误差确定迭代是否终止
            
            %% 计算原始迭代W，B
            %% 误差反传，计算 dW, dB, 并迭代 W, B
            e{end} = L{2}.*H{end,3};
            W{end,2} = -Train_rate*H{end-1,1}'*e{end};
            Bias{end,2} = Train_rate*e{end};
            for j = length(HID)-1:-1:1
                e{j} = e{j+1}*(W{j+1,1})'.*H{j+1,3};
                W{j,2} = -Train_rate*H{j,1}'*e{j};
                Bias{j,2} = Train_rate*e{j};
            end

            for j = 1:length(HID)
                %% 权重及偏置量代换
                W{j,1} = W{j,1} + W{j,2};
                Bias{j,1} = Bias{j,1} + Bias{j,2};
                W_annealing{j,1} = W{j,1} + (1 - Train_rate)*W{j,2};
                Bias_annealing{j,1} = Bias{j,1} + (1 - Train_rate)*Bias{j,2};
            end
            %% 梯度下降求 dW, dB, dL
            H{1,1} = Data0;
            H{1,2} = Data0;
            for j = 1:length(HID)
                W_data = W{j,1};
                H_data = H{j,2};
                H{j+1,1} = H_data*W_data - Bias{j};
                [H{j+1,2}, H{j+1,3}] = ExcFunction_dF(H{j+1,1},ExcFunc);
            end
            [L{1}, L{2}] = LossFunction(H{end,2}, real_Label0, LossFunc(1), LossFunc(2), 1.5);
            
            %% 退火求 dW, dB, dL
            H_annealing{1,1} = Data0;
            H_annealing{1,2} = Data0;
            for j = 1:length(HID)
                W_data = W_annealing{j,1};
                H_data = H_annealing{j,2};
                H_annealing{j+1,1} = H_data*W_data - Bias_annealing{j};
                [H_annealing{j+1,2}, H_annealing{j+1,3}] = ExcFunction_dF(H_annealing{j+1,1},ExcFunc);
            end
            [L_annealing{1}, L_annealing{2}] = LossFunction(H_annealing{end,2}, real_Label0, LossFunc(1), LossFunc(2), 1.5);

            %% 确定是否退火
            Accept = SimulatedAnnealing(L_annealing{1},L{1},T);

            %% 不接受该次迭代，替换为 (1 - Train_rate) 迭代
            if Accept == 0
                W = W_annealing;
                Bias = Bias_annealing;
                continue;
            end
        end
    end
    net = {Layer,W,Bias,ExcFunc,sum(abs(L{2})),T,LPlot};
end
