function [results,tag_test_data]=KNN_data(set_data,set_tags_data)
%for data 60,20,20:
train_data=[];
tag_train_data=[];
test_data=[];
tag_test_data=[];

idx = randperm(371);%generate random vector of size 371
for j=1:1:260%training
    train_data=[train_data; set_data(idx(j),:)];
    tag_train_data=[tag_train_data;set_tags_data(idx(j),:)];
           
end




for j=261:1:371% test
    test_data=[test_data; set_data(idx(j),:)];
    tag_test_data=[tag_test_data;set_tags_data(idx(j),:)];
        
        
end


tot_sum_data=[];
for i=1:100
    mdl=fitcknn(train_data,tag_train_data,'NumNeighbors',i,'Standardize',1);

    results=predict(mdl,test_data);
    sum=0;
    for j=1:111
   
        if(strcmp(results(j,1),tag_test_data(j,1)))
            sum=sum+1;
        end
    end
    tot_sum_data=[ tot_sum_data,sum];
    
end

bigger=0;
flag=0;
for i=1:100
 if(tot_sum_data(i)>bigger)
     bigger=tot_sum_data(i);
     flag=i;
 end
end
mdl=fitcknn(train_data,tag_train_data,'NumNeighbors',flag,'Standardize',1);
results=predict(mdl,test_data);
sum=0;
for j=1:111
    if(strcmp(results(j,1),tag_test_data(j,1)))
            sum=sum+1;
    end
end


answer=['Data: The KNN predicts right ',num2str(sum),' from test set of 111 samples. ',  'number of neighbors: ',num2str(flag) ];
disp(answer);
%{
tot_sum_data=0;
for i=1:100
    mdl=fitcknn(train_data,tag_train_data);
    results=predict(mdl,test_data);
    sum=0;
    for j=1:74
   
        if(strcmp(results(j,1),tag_test_data(j,1)))
            sum=sum+1;
        end
    end
    tot_sum_data=tot_sum_data+sum;
end
tot_sum_data=tot_sum_data/100;%how mach it predict right from 71 samples of test
answer=['Data: The KNN predicts right ',num2str(tot_sum_data),' from test set of 74 samples' ];
disp(answer)
%}
end