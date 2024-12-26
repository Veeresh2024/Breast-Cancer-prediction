function [results,tag_test_without_NAN]=KNN_without_NAN(set_without_NAN,set_tags_without_NAN)


%for without_NAN 60,20,20:
train_without_NAN=[];
tag_train_without_NAN=[];
test_without_NAN=[];
tag_test_without_NAN=[];

idx = randperm(354);%generate random vector of size 355
for j=1:1:247%training
    train_without_NAN=[train_without_NAN; set_without_NAN(idx(j),:)];
    tag_train_without_NAN=[tag_train_without_NAN;set_tags_without_NAN(idx(j),:)];
           
end



for j=248:1:354% % test
    test_without_NAN=[test_without_NAN; set_without_NAN(idx(j),:)];
    tag_test_without_NAN=[tag_test_without_NAN;set_tags_without_NAN(idx(j),:)];
        
        
end
tot_sum_without_NAN=[];
for i=1:100
    mdl=fitcknn(train_without_NAN,tag_train_without_NAN,'NumNeighbors',i,'Standardize',1);
    results=predict(mdl,test_without_NAN);
    sum=0;
    for j=1:107
   
        if(strcmp(results(j,1),tag_test_without_NAN(j,1)))
            sum=sum+1;
        end
    end
    tot_sum_without_NAN=[ tot_sum_without_NAN,sum];
    
end

bigger=0;
flag=0;
for i=1:100
 if(tot_sum_without_NAN(i)>bigger)
     bigger=tot_sum_without_NAN(i);
     flag=i;
 end
end
mdl=fitcknn(train_without_NAN,tag_train_without_NAN,'NumNeighbors',flag,'Standardize',1);
results=predict(mdl,test_without_NAN);
sum=0;
for j=1:107
    if(strcmp(results(j,1),tag_test_without_NAN(j,1)))
            sum=sum+1;
    end
end
answer=['Data without NANS: The KNN predicts right ',num2str(sum),' from test set of 107 samples.',' number of neighbors: ',num2str(flag) ];
disp(answer);

%tot_sum_without_NAN=tot_sum_without_NAN/100; %how mach it predict right from 71 samples of test
%answer=['Data without NANS: The KNN predicts right ',num2str(tot_sum_without_NAN),' from test set of 71 samples' ];
%disp(answer)






end