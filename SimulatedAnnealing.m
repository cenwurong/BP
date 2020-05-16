%% 模拟退火算法，根据误差函数确定是否接受状态改变
function Accept = SimulatedAnnealing(L1,L2,T)
    l1 = sum(abs(L1));
    l2 = sum(abs(L2));
    temp1 = rand();
    if l1 > l2
        Accept = 1;
        return;
    else
        temp2 = exp(-(l2-l1)/T);
        if temp1 < temp2
            Accept = 1;
            return;
        end
    end
    
    Accept = 0;
    
end