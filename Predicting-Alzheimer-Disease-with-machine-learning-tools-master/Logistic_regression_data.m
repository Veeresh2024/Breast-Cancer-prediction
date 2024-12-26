function [ results_new_LR,results,tag_test_data]=Logistic_regression_data(set_data,set_tags_data, set_data_new)

t_ = categorical(set_tags_data);




 

%for data 70,30:
train_data=[];
tag_train_data=[];
test_data=[];
tag_test_data=[];

idx = randperm(371);%generate random vector of size 371
for j=1:1:260%training
    train_data=[train_data; set_data(idx(j),:)];
    tag_train_data=[tag_train_data;t_(idx(j),:)];
           
end




for j=261:1:371% test
    test_data=[test_data; set_data(idx(j),:)];
    tag_test_data=[tag_test_data;t_(idx(j),:)];
        
        
end

 [B,dev] = mnrfit(train_data,tag_train_data);
 results = mnrval(B,test_data);
 results_new_LR=mnrval(B,set_data_new);
 sum=0;
for j=1:111
    if((results(j,1)>results(j,2))&&(tag_test_data(j,1)=='Demented'  ))
        sum=sum+1;
    elseif((results(j,2)>results(j,1))&&(tag_test_data(j,1)=='Nondemented'  ))
        sum=sum+1;

    end
end
tot_sum_data= sum;
answer=['Data: The Logistic_regression predicts right ',num2str(sum),' from test set of 111 samples' ];
disp(answer);
end