%% BP 测试，返回对应标签，[1,0;0,1;1,0]
function Label = BPTest(net,Data)
%% 参数检查，暂无
    if nargin < 2 
        fprintf("Please input right net and data for test!");
    end
    Layer = net{1};
    W = net{2};
    Bias = net{3};
    ExcFunc = net{4};
    if Layer(1) ~= length(Data(1,:))
        fprintf("Error:input data's dimension not match input layer nodes!");
    end

    Label = zeros(length(Data(:,1)),Layer(end));
    for i =1:length(Label(:,1))
        Data0 = Data(i,:);
        hin = Data0;
        for j = 1:length(W)
            hout = hin*W{j,1} - Bias{j,1};
            [fhout,~] = ExcFunction_dF(hout, ExcFunc);
            hin = fhout;
        end
        temp = max(fhout);
        temp_fhout = fhout;
        temp_fhout(fhout~=temp) = 0;
        temp_fhout(fhout==temp) = 1;
        Label(i,:) = temp_fhout;
    end
end