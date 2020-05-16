%% 标签转化
%% 多分类问题转化为高维0,1问题，如1/2/3分类问题，转化为三维(1,0,0）,(0,1,0),(0,0,1)
% real_label = ["A","B","C",1];
% c_label = Categorical1(real_label);
function C_label = Categorical(real_label)
    if nargin == 0 
        fprintf("Please input right label to change!");
        return;
    elseif nargin > 1
        fprintf("Input data so many parameters!");
        return;
    end
    temp1 = unique(real_label);
    N = length(temp1);
    temp_label = zeros(length(real_label),1);
    for i = 1:N
        index = real_label == temp1(i);
        temp_label(index) = i;
    end
    
    L = length(temp_label);
    C_label = zeros(L,N);
    for i = 1:L
        C_label(i,temp_label(i)) = 1;
    end
end