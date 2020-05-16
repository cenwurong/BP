%% BP ������ѵ����
%% ���������ת��Ϊ��ά0,1���⣬��1/2/3�������⣬ת��Ϊ��ά(1,0,0��,(0,1,0),(0,0,1)
%% ����ٶ�label���Ѿ�ת������
%% ��ʧ����һ���ý�����
function net = BPTrain(Data,Label,hid,Train_rate,ExcFunc,LossFunc)
% ExcFunc: "sigmoid" or "tanh" or "ReLU" or "PReLU" or "ELU"
% LossFunction: ["Regression" "MSE"] or ["Regression" "MAE"] or ["Regression" "Huber"]
%            or ["Classification" "Logistic_Loss"] or ["Classification" "Cross_Entropy_Loss"]
%            or ["Classification" "Hinge_Loss"] or ["Classification" "exp_Loss"] 
%            or ["Classification" "Modified_Huber_Loss"]
    %% �������
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
    %% ����������
    
    %% ���������������
    temp1 = length(unique(Label,'rows'));
    HID = [hid temp1]; %% ����� + ���ز� + ����� - 1


    W = cell(length(HID),2);  %% Ȩ�����, W, dW�� ����� + ���ز� + ����� - 1
    H = cell(1 + length(HID),3);  %% �����������, H_in=XW-B, H_out=f(H_in), f'  ����� + ���ز� + �����
    Bias = cell(length(HID),2); %% ƫ�������, Bias, dBias   ����� + ���ز� + ����� - 1
    L = cell(1,2);           %% ��ʧ�������, L, dL  �����
    L_annealing = cell(1,2);           %% ��ʧ�������, L, dL  �����
    e = cell(length(HID),1);  %%  ����� + ���ز� + ����� - 1

    Layer = [length(Data(1,:)) HID];

    H{1,1} = zeros(size(Data(1,:)));
    H{1,2} = zeros(size(Data(1,:)));
    for i = 1:length(HID)
        H{i+1,1} = zeros(1,HID(i));
        W{i,1} = 2*rand(length(H{i,1}),length(H{i+1,1})) - 1;
        Bias{i,1} = 2*rand(1,HID(i)) - 1;
    end
    %% �˻��¶� ��ʼ��
    Tmax = 1000;
    Tmin = 0.01;
    T = Tmax;
    W_annealing = W;
    Bias_annealing = Bias;
    H_annealing = H;
    
    NLoop = 0;  %%% ѭ����������
    Delta = 0.00001;  %%% ����ʧ���仯 dL
    L_err = 100;
% % %     while (T > Tmin) && (NLoop < 10000)
    LPlot = [];
    while T > Tmin
%     while NLoop < 10000
        %%% �˻��¶�С������˻��¶� �� ����ʧ���仯С��Delta �� ѭ���������ѭ������NLoop ����ѭ��ֹͣ
% % %         if L_err < Delta
% % %             break;
% % %         end
        NLoop = NLoop + 1;
        T = 0.99*T; %% 
        L_err = 0;
        for i_event = 1:length(Data(:,1))
            %% ������Ϊ��λѭ��
            Data0 = Data(i_event,:);
            real_Label0 = Label(i_event,:);

            H{1,1} = Data0;
            H{1,2} = Data0;
            for j = 1:length(HID)
                %% ����Ȩ�ؼ�����ڵ���
                W_data = W{j,1};
                H_data = H{j,2};
                H{j+1,1} = H_data*W_data - Bias{j};
                [H{j+1,2}, H{j+1,3}] = ExcFunction_dF(H{j+1,1},ExcFunc);
            end
            %% ��һ������ʧ�������
            [L{1}, L{2}] = LossFunction(H{end,2}, real_Label0, LossFunc(1), LossFunc(2), 1.5);
            L_err = L_err + sum(abs(L{2}));
            LPlot = [LPlot sum(L{1})];
            %% ������ʧ�������ȷ�������Ƿ���ֹ
            
            %% ����ԭʼ����W��B
            %% ���������� dW, dB, ������ W, B
            e{end} = L{2}.*H{end,3};
            W{end,2} = -Train_rate*H{end-1,1}'*e{end};
            Bias{end,2} = Train_rate*e{end};
            for j = length(HID)-1:-1:1
                e{j} = e{j+1}*(W{j+1,1})'.*H{j+1,3};
                W{j,2} = -Train_rate*H{j,1}'*e{j};
                Bias{j,2} = Train_rate*e{j};
            end

            for j = 1:length(HID)
                %% Ȩ�ؼ�ƫ��������
                W{j,1} = W{j,1} + W{j,2};
                Bias{j,1} = Bias{j,1} + Bias{j,2};
                W_annealing{j,1} = W{j,1} + (1 - Train_rate)*W{j,2};
                Bias_annealing{j,1} = Bias{j,1} + (1 - Train_rate)*Bias{j,2};
            end
            %% �ݶ��½��� dW, dB, dL
            H{1,1} = Data0;
            H{1,2} = Data0;
            for j = 1:length(HID)
                W_data = W{j,1};
                H_data = H{j,2};
                H{j+1,1} = H_data*W_data - Bias{j};
                [H{j+1,2}, H{j+1,3}] = ExcFunction_dF(H{j+1,1},ExcFunc);
            end
            [L{1}, L{2}] = LossFunction(H{end,2}, real_Label0, LossFunc(1), LossFunc(2), 1.5);
            
            %% �˻��� dW, dB, dL
            H_annealing{1,1} = Data0;
            H_annealing{1,2} = Data0;
            for j = 1:length(HID)
                W_data = W_annealing{j,1};
                H_data = H_annealing{j,2};
                H_annealing{j+1,1} = H_data*W_data - Bias_annealing{j};
                [H_annealing{j+1,2}, H_annealing{j+1,3}] = ExcFunction_dF(H_annealing{j+1,1},ExcFunc);
            end
            [L_annealing{1}, L_annealing{2}] = LossFunction(H_annealing{end,2}, real_Label0, LossFunc(1), LossFunc(2), 1.5);

            %% ȷ���Ƿ��˻�
            Accept = SimulatedAnnealing(L_annealing{1},L{1},T);

            %% �����ܸôε������滻Ϊ (1 - Train_rate) ����
            if Accept == 0
                W = W_annealing;
                Bias = Bias_annealing;
                continue;
            end
        end
    end
    net = {Layer,W,Bias,ExcFunc,sum(abs(L{2})),T,LPlot};
end
