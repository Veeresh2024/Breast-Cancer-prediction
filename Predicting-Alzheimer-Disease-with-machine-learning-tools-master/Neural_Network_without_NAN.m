function [results,tag_test_without_NAN]=Neural_Network_without_NAN(set_without_NAN,set_tags_without_NAN)
set_tags_without_NAN1 = categorical(set_tags_without_NAN);
set_tags_without_NAN1 = dummyvar(set_tags_without_NAN1);
set_tags_without_NAN1=set_tags_without_NAN1';
set_without_NAN=set_without_NAN';
train_without_NAN=[];
tag_train_without_NAN=[];
test_without_NAN=[];
tag_test_without_NAN=[];

idx = randperm(354);%generate random vector of size 371
for j=1:1:247%training
    train_without_NAN=[train_without_NAN, set_without_NAN(:,idx(j))];
    tag_train_without_NAN=[tag_train_without_NAN,set_tags_without_NAN1(:,idx(j))];
           
end


for j=248:1:354% test
    test_without_NAN=[test_without_NAN,set_without_NAN(:,idx(j))];
    tag_test_without_NAN=[tag_test_without_NAN,set_tags_without_NAN1(:,idx(j))];   
end

results=zeros(2,107);
for i = 1:1:100
    net = patternnet(10);
    [net, tr]=train(net,train_without_NAN, tag_train_without_NAN);
    tag_test_without_NAN_try = net(test_without_NAN);
    results=results+tag_test_without_NAN_try;
end
results=results/100;

sum=0;
for j=1:107
    if((results(1,j)>results(2,j))&&(tag_test_without_NAN(1,j)==1  ))
        sum=sum+1;
    elseif((results(2,j)>results(1,j))&&(tag_test_without_NAN(2,j)==1 ))
        sum=sum+1;
    end
end
tot_sum_without_NAN= sum;
answer=['Data without NANS: The neural network predicts right ',num2str(sum),' from test set of 107 samples. ' ];
disp(answer);
end
