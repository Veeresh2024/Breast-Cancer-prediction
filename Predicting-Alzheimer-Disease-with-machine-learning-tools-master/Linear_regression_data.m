function [results,tag_test_data]=Linear_regression_data(set_data,set_tags_data)

t=zeros(371,1);
for i=1:371%making the tags matrix
    if(strcmp(set_tags_data(i,1) ,'Demented'))
       
        t(i,1)=1;
    elseif(strcmp(set_tags_data(i,1) ,'Nondemented'))
        t(i,1)=2;
     
        
    end
end




 

%for data 70,30:
train_data=[];
tag_train_data=[];
test_data=[];
tag_test_data=[];

idx = randperm(371);%generate random vector of size 354
for j=1:1:260%training
    train_data=[train_data; set_data(idx(j),:)];
    tag_train_data=[tag_train_data;t(idx(j),:)];
           
end




for j=261:1:371% test
    test_data=[test_data; set_data(idx(j),:)];
    tag_test_data=[tag_test_data;t(idx(j),:)];
        
        
end

dsa = dataset(train_data(1:260,1), train_data(1:260,2) ,train_data(1:260,3),train_data(1:260,4),train_data(1:260,5),train_data(1:260,6) ,train_data(1:260,7) ,train_data(1:260,8) ,train_data(1:260,9),train_data(1:260,10),tag_train_data,'VarNames',{'MR Delay','Age','EDUC','SES','MMSE','CDR','eTIV','nWBV','ASF','M/F','group'});%create a database
lm = fitlm(dsa);
results=predict(lm,test_data(:,1:10));
sum=0;
for j=1:111
    if(round(results(j,1))==tag_test_data(j,1))
        sum=sum+1;
    end
end

answer=['Data: The Linear_regression predicts right ',num2str(sum),' from test set of 111 samples' ];
disp(answer);


tot_sum_data=sum;
end