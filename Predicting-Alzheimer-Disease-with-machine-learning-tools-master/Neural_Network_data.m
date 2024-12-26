function [results_new_nn,results,tag_test_data]=Neural_Network_data(set_data,set_tags_data,set_data_new)
set_tags_data1 = categorical(set_tags_data);
set_tags_data1 = dummyvar(set_tags_data1);
set_tags_data1=set_tags_data1';
set_data=set_data';
set_data_new=set_data_new';
train_data=[];
tag_train_data=[];
test_data=[];
tag_test_data=[];

idx = randperm(371);%generate random vector of size 371
for j=1:1:260%training
    train_data=[train_data, set_data(:,idx(j))];
    tag_train_data=[tag_train_data,set_tags_data1(:,idx(j))];
           
end


for j=261:1:371% test
    test_data=[test_data, set_data(:,idx(j))];
    tag_test_data=[tag_test_data,set_tags_data1(:,idx(j))];   
end

results=zeros(2,111);
results_new_nn=zeros(2,216);
for i = 1:1:100
    net = patternnet(10);
    [net, tr]=train(net,train_data, tag_train_data);
    tag_test_data_try = net(test_data);
    tag_new_data=net(set_data_new);
    results_new_nn=results_new_nn+ tag_new_data;
    results=results+tag_test_data_try;
end
results=results/100;
results_new_nn= results_new_nn/100;

sum=0;
for j=1:111
    if((results(1,j)>results(2,j))&&(tag_test_data(1,j)==1  ))
        sum=sum+1;
    elseif((results(2,j)>results(1,j))&&(tag_test_data(2,j)==1 ))
        sum=sum+1;
    end
end
tot_sum_data= sum;
answer=['Data: The neural network predicts right ',num2str(sum),' from test set of 111 samples. ' ];
disp(answer);
end