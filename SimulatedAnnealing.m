%% ģ���˻��㷨����������ȷ���Ƿ����״̬�ı�
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