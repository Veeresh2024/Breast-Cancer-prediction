warning off;
filename = 'oasis_longitudinal.csv';
learn_fileName='oasis_cross-sectional.csv';
[~,~,data_new] =xlsread(learn_fileName);
[~,~,data] =xlsread(filename);

without_NAN=data;
without_NAN(any(cellfun(@(x) any(isnan(x)),without_NAN),2),:) = [];


%=================================   
h = figure ('Color', [1 1 1]);
s1 = plot(cell2mat(without_NAN(2:end,9)),cell2mat(without_NAN(2:end,10)),'k+');
set(s1, 'MarkerSize', 8, 'LineWidth', 2);
%%% regression line
hold on
l = lsline ;
set(l,'LineWidth', 2)
%%% axis display 
xlabel('EDUC')
ylabel('SES')
set(gca, 'YMinorTic','on','XMinorTick','on')
hold off
%===================================

EDUC_SES= [cell2mat(without_NAN(2:end,9)),cell2mat(without_NAN(2:end,10))];
EDUC_SES=sortrows(EDUC_SES,1);
[mea, med] = grpstats(EDUC_SES(:,2), EDUC_SES(:,1), {'mean', @median});
EDUC_MeanSES=[unique(EDUC_SES(:,1)),med];

for i=1:372
    if(isnan(data{i,10}))
        for j=1:12
            if (data{i,9}==EDUC_MeanSES(j,1))
                 data{i,10}=EDUC_MeanSES(j,2);
            end
        end
       
    end
end

data_new(:,1)=[];%id
data_new(:,3)=[];%%delete the Hand->right for every one

MF_data_new=categorical(data_new(2:217,2));%change M,F to categorical
MF_data_new=dummyvar(MF_data_new);

data(:,7)=[];
without_NAN(:,7)=[];%delete the Hand->right for every one
MF_data=categorical(data(2:372,6));%change M,F to categorical
MF_data=dummyvar(MF_data);
MF_without_NAN=categorical(without_NAN(2:355,6));%change M,F to categorical
MF_without_NAN=dummyvar(MF_without_NAN);

group_data=categorical(data(2:372,3));%change group to categorical
group_data=dummyvar(group_data);

group_without_NAN=categorical(without_NAN(2:355,3));%change group to categorical
group_without_NAN=dummyvar(group_without_NAN);

 set_data_new=[ data_new{2:217,1}; data_new{2:217,3} ;data_new{2:217,4} ;data_new{2:217,5} ;data_new{2:217,6};data_new{2:217,7};data_new{2:217,8};data_new{2:217,9};data_new{2:217,10}]';
 set_data = [ data{2:372,5}; data{2:372,7} ;data{2:372,8} ;data{2:372,9} ;data{2:372,10};data{2:372,11};data{2:372,12};data{2:372,13};data{2:372,14}]';
 set_data=[set_data MF_data];
 set_data_new=[set_data_new MF_data_new];
 set_tags_data=data(2:372,3);
 
 
 set_without_NAN = [ without_NAN{2:355,5}; without_NAN{2:355,7} ;without_NAN{2:355,8} ;without_NAN{2:355,9} ;without_NAN{2:355,10} ;without_NAN{2:355,11} ;without_NAN{2:355,12} ;without_NAN{2:355,13} ;without_NAN{2:355,14}]';
 set_without_NAN=[set_without_NAN MF_without_NAN];
 set_tags_without_NAN=without_NAN(2:355,3);

 %bar graph
 bar_data=[set_tags_data,num2cell(set_data(:,10))];
 F_Demented=0;
 F_Nondemented=0;
 M_Demented=0;
 M_Nondemented=0;
 for i=1:371
     if(strcmp(bar_data(i,1),'Demented')&&(set_data(i,10)==1))
         F_Demented= F_Demented+1;
     elseif(strcmp(bar_data(i,1),'Nondemented')&&(set_data(i,10)==1))
         F_Nondemented=F_Nondemented+1;
     elseif(strcmp(bar_data(i,1),'Demented')&&(set_data(i,10)==0))
         M_Demented=M_Demented+1;
     elseif(strcmp(bar_data(i,1),'Nondemented')&&(set_data(i,10)==0))
         M_Nondemented=M_Nondemented+1;
     end
 end
 figure%The above graph indicates that men are more likely with dementia than women.
 y = [ F_Nondemented  M_Nondemented;  F_Demented   M_Demented];
 bar(y);
 set(gca,'XTickLabel',{'Nondemented','Demented'})
 legend('F','M')
 title('Gender and Demented rate')
 xlabel('Group')
 ylabel('rate')
 clr = [1 0 0;
   0 0 1];
colormap(clr);


%MMSE
group_data=categorical(data(2:372,3));%change group to categorical
group_data=dummyvar(group_data);

GROUP_MMSE= [group_data(:,1),cell2mat(data(2:end,10))];
MMSE_sort=sortrows(GROUP_MMSE,2);
Nondemented=zeros(31,2);
Score_MMSE=(0:30)';
Nondemented(:,1)=Score_MMSE;
Demented=zeros(31,2);
Demented(:,1)=Score_MMSE;


for i=1:1:371
    for j=1:1:31
        if(MMSE_sort(i,2)==Nondemented(j,1)&&MMSE_sort(i,1)==0)
            Nondemented(j,2)=Nondemented(j,2)+1;
        end
        if(MMSE_sort(i,2)==Demented(j,1)&&MMSE_sort(i,1)==1)
            Demented(j,2)=Demented(j,2)+1;
        end
    end
end
figure%The chart shows Nondemented group got much more higher MMSE scores than Demented group.
hold on
plot( Nondemented(:,1), Nondemented(:,2),'b')
plot( Demented(:,1), Demented(:,2),'r')
xlabel('MMSE Score');
ylabel('number');
title('MMSE level')
legend('Nondemented','Demented')
hold off


%nWBV
group_data=categorical(data(2:372,3));%change group to categorical
group_data=dummyvar(group_data);

GROUP_nWBV= [group_data(:,1),cell2mat(data(2:end,13))];
nWBV_sort=sortrows(GROUP_nWBV,2);
Nondemented=zeros(7,2);
Score_nWBV=zeros(7,2);
Score_nWBV=[0.60,0.65,0.70,0.75,0.80,0.85,0.90]';
Nondemented(:,1)=Score_nWBV;
Demented=zeros(7,2);
Demented(:,1)=Score_nWBV;

for i=1:1:371
    for j=1:1:7
        if(nWBV_sort(i,2)<=Nondemented(j,1)&&nWBV_sort(i,2)>Nondemented(j-1,1)&&nWBV_sort(i,1)==0)
            Nondemented(j,2)=Nondemented(j,2)+1;
        end
        if(nWBV_sort(i,2)<=Demented(j,1)&&nWBV_sort(i,2)>Demented(j-1,1)&&nWBV_sort(i,1)==1)
            Demented(j,2)=Demented(j,2)+1;
        end
    end
end
figure%The chart indicates that Nondemented group has higher brain volume ratio than Demented group. This is assumed to be because the diseases affect the brain to be shrinking its tissue.
hold on
plot( Nondemented(:,1), Nondemented(:,2),'b')
plot( Demented(:,1), Demented(:,2),'r')
xlabel('nWBV level');
ylabel('number');
title('nWBV level')
legend('Nondemented','Demented')
hold off



 
 
 
%KNN


 [results,tag_test_without_NAN]=KNN_without_NAN(set_without_NAN,set_tags_without_NAN);
 [confMat,order] = confusionmat(tag_test_without_NAN,results);
 Recall_knn_without_NAN=confMat(1,1)/(confMat(1,1)+confMat(2,1));%the higher-the better preformance
 precision_knn_without_NAN=confMat(1,1)/(confMat(1,1)+confMat(1,2));%the higher-the better preformance
 F1_knn_without_NAN=(2*Recall_knn_without_NAN*precision_knn_without_NAN)/(Recall_knn_without_NAN+precision_knn_without_NAN);%the higher-the better preformance
 
[results,tag_test_data]=KNN_data(set_data,set_tags_data);
[confMat,order] = confusionmat(tag_test_data,results);
Recall_knn_data=confMat(1,1)/(confMat(1,1)+confMat(2,1));%the higher-the better preformance
precision_knn_data=confMat(1,1)/(confMat(1,1)+confMat(1,2));%the higher-the better preformance
F1_knn_data=(2*Recall_knn_data*precision_knn_data)/(Recall_knn_data+precision_knn_data);%the higher-the better preformance

%================================================================
false_negative=0;
false_positive=0;
true_negative=0;
true_positive=0;
for i=1:111
    if(strcmp(tag_test_data(i),results(i))&&strcmp(tag_test_data(i),'Demented'))
        true_positive=true_positive+1;
    elseif(strcmp(tag_test_data(i),results(i))&&strcmp(tag_test_data(i),'Nondemented'))
        true_negative=true_negative+1;
    end
end
%=================================================================



%Linear_regression
[results,tag_test_without_NAN]=Linear_regression_without_NAN(set_without_NAN,set_tags_without_NAN);
[confMat,order] = confusionmat(tag_test_without_NAN,round(results));
Recall_linear_r_without_NAN=confMat(1,1)/(confMat(1,1)+confMat(2,1));%the higher-the better preformance
precision_linear_r_without_NAN=confMat(1,1)/(confMat(1,1)+confMat(1,2));%the higher-the better preformance
F1_linear_r_without_NAN=(2*Recall_linear_r_without_NAN*precision_linear_r_without_NAN)/(Recall_linear_r_without_NAN+precision_linear_r_without_NAN);%the higher-the better preformance


[results,tag_test_data]=Linear_regression_data(set_data,set_tags_data);
[confMat,order] = confusionmat(tag_test_data,round(results));
Recall_linear_r_data=confMat(1,1)/(confMat(1,1)+confMat(2,1));%the higher-the better preformance
precision_linear_r_data=confMat(1,1)/(confMat(1,1)+confMat(1,2));%the higher-the better preformance
F1_linear_r_data=(2*Recall_linear_r_data*precision_linear_r_data)/(Recall_linear_r_data+precision_linear_r_data);%the higher-the better preformance


%Logistic_regression

[ results,tag_test_without_NAN]=Logistic_regression_without_NAN(set_without_NAN,set_tags_without_NAN);
TP=0;
TN=0;
FP=0;
FN=0;
for j=1:107
    if((results(j,1)>results(j,2))&&(tag_test_without_NAN(j,1)=='Demented'  ))
        TP=TP+1;
    elseif((results(j,2)>results(j,1))&&(tag_test_without_NAN(j,1)=='Nondemented'  ))
        TN=TN+1;
    elseif((results(j,1)>results(j,2))&&(tag_test_without_NAN(j,1)=='Nondemented'  ))
        FP=FP+1;
     elseif((results(j,2)>results(j,1))&&(tag_test_without_NAN(j,1)=='Demented'  ))
        FN=FN+1;

    end
end
Recall_logistic_r_without_NAN=TP/(TP+FN);
Precision_logistic_r_without_NAN=TP/(TP+FP);
F1_logistic_r_without_NAN=(2*Recall_logistic_r_without_NAN*Precision_logistic_r_without_NAN)/(Recall_logistic_r_without_NAN+Precision_logistic_r_without_NAN);

[results_new_LR,results,tag_test_data]=Logistic_regression_data(set_data,set_tags_data, set_data_new);
TP=0;
TN=0;
FP=0;
FN=0;
for j=1:111
    if((results(j,1)>results(j,2))&&(tag_test_data(j,1)=='Demented'  ))
        TP=TP+1;
    elseif((results(j,2)>results(j,1))&&(tag_test_data(j,1)=='Nondemented'  ))
        TN=TN+1;
    elseif((results(j,1)>results(j,2))&&(tag_test_data(j,1)=='Nondemented'  ))
        FP=FP+1;
     elseif((results(j,2)>results(j,1))&&(tag_test_data(j,1)=='Demented'  ))
        FN=FN+1;

    end
end
Recall_logistic_r_data=TP/(TP+FN);
Precision_logistic_r_data=TP/(TP+FP);
F1_logistic_r_data=(2*Recall_logistic_r_data*Precision_logistic_r_data)/(Recall_logistic_r_data+Precision_logistic_r_data);




%NN 

[results,tag_test_without_NAN]=Neural_Network_without_NAN(set_without_NAN,set_tags_without_NAN);
TP=0;
TN=0;
FP=0;
FN=0;
for j=1:107
    if((results(1,j)>results(2,j))&&(tag_test_without_NAN(1,j)==1  ))
        TP=TP+1;
    elseif((results(2,j)>results(1,j))&&(tag_test_without_NAN(2,j)==1 ))
        TN=TN+1;
     elseif((results(1,j)>results(2,j))&&(tag_test_without_NAN(2,j)==1 ))
        FP=FP+1;
     elseif((results(2,j)>results(1,j))&&(tag_test_without_NAN(1,j)==1 ))
        FN=FN+1;
    end
end

Recall_NN_without_NAN=TP/(TP+FN);
Precision_NN_without_NAN=TP/(TP+FP);
F1_NN_without_NAN=(2*Recall_NN_without_NAN*Precision_NN_without_NAN)/(Recall_NN_without_NAN+Precision_NN_without_NAN);


[results_new_nn,results,tag_test_data]=Neural_Network_data(set_data,set_tags_data,set_data_new);

TP=0;
TN=0;
FP=0;
FN=0;
for j=1:111
    if((results(1,j)>results(2,j))&&(tag_test_data(1,j)==1  ))
        TP=TP+1;
    elseif((results(2,j)>results(1,j))&&(tag_test_data(2,j)==1 ))
        TN=TN+1;
     elseif((results(1,j)>results(2,j))&&(tag_test_data(2,j)==1 ))
        FP=FP+1;
     elseif((results(2,j)>results(1,j))&&(tag_test_data(1,j)==1 ))
        FN=FN+1;
    end
end

Recall_NN_data=TP/(TP+FN);
Precision_NN_data=TP/(TP+FP);
F1_NN_data=(2*Recall_NN_data*Precision_NN_data)/(Recall_NN_data+Precision_NN_data);


%SVM 

[indx_Demented,indx_Nondemented,label_Demented,label_Nondemented]=SVM_data(set_data,set_tags_data);
TP=0;
TN=0;
FP=0;
FN=0;
for i=1:111
        if(indx_Demented(i)==1&&label_Demented(i)==1)
            TP=TP+1;
       
        elseif(indx_Demented(i)==0&&label_Demented(i)==0)
             TN=TN+1;
        elseif(indx_Demented(i)==1&&label_Demented(i)==0)
             FN=FN+1;
        elseif(indx_Demented(i)==0&&label_Demented(i)==1)
             FP=FP+1;
        end
end
 
Recall_SVM_data=TP/(TP+FN);
Precision_SVM_data=TP/(TP+FP);
F1_SVM_data=(2*Recall_SVM_data*Precision_SVM_data)/(Recall_SVM_data+Precision_SVM_data);


[indx_Demented,indx_Nondemented,label_Demented,label_Nondemented]=SVM_without_NAN(set_without_NAN,set_tags_without_NAN);

TP=0;
TN=0;
FP=0;
FN=0;
for i=1:107
        if(indx_Demented(i)==1&&label_Demented(i)==1)
            TP=TP+1;
       
        elseif(indx_Demented(i)==0&&label_Demented(i)==0)
             TN=TN+1;
        elseif(indx_Demented(i)==1&&label_Demented(i)==0)
             FN=FN+1;
        elseif(indx_Demented(i)==0&&label_Demented(i)==1)
             FP=FP+1;
        end
end
 
Recall_SVM_without_NAN=TP/(TP+FN);
Precision_SVM_without_NAN=TP/(TP+FP);
F1_SVM_without_NAN=(2*Recall_SVM_without_NAN*Precision_SVM_without_NAN)/(Recall_SVM_without_NAN+Precision_SVM_without_NAN);


%decision tree

[results_new_decision,Tag_new,tag_test_data]=Decision_tree_data(set_data,set_tags_data,set_data_new);
[confMat,order] = confusionmat(tag_test_data,Tag_new);
Recall_tree_data=confMat(1,1)/(confMat(1,1)+confMat(2,1));%the higher-the better preformance
precision_tree_data=confMat(1,1)/(confMat(1,1)+confMat(1,2));%the higher-the better preformance
F1_tree_data=(2*Recall_tree_data*precision_tree_data)/(Recall_tree_data+precision_tree_data);%the higher-the better preformance

[Tag_new,tag_test_without_NAN]=Decision_tree_without_NAN(set_without_NAN,set_tags_without_NAN);
[confMat,order] = confusionmat(tag_test_without_NAN,Tag_new);
Recall_tree_without_NAN=confMat(1,1)/(confMat(1,1)+confMat(2,1));%the higher-the better preformance
precision_tree_without_NAN=confMat(1,1)/(confMat(1,1)+confMat(1,2));%the higher-the better preformance
F1_tree_without_NAN=(2*Recall_tree_without_NAN*precision_tree_without_NAN)/(Recall_tree_without_NAN+precision_tree_without_NAN);%the higher-the better preformance


Demented=zeros(216,2);
Nondemented=zeros(216,2);
for t=1:216
    if(strcmp(results_new_decision(t),'Demented'))
        Demented(t)=Demented(t)+1;
    
    end
    if(results_new_LR(t,1)>=results_new_LR(t,2))
         Demented(t)=Demented(t)+1;
    end
    if(results_new_nn(1,t)>=results_new_nn(2,t))
        Demented(t)=Demented(t)+1;
    end

end
tags_new=zeros(216,1);
for i=1:216
    if(Demented(i,1)>=2)
        tags_new(i)=1;
    end
end

%bar graph
F_Demented=0;
F_Nondemented=0;
M_Demented=0;
M_Nondemented=0;
 for i=1:216
     if((tags_new(i)==1)&&(MF_data_new(i,1)==1))
         F_Demented= F_Demented+1;
     elseif((tags_new(i)==0)&&(MF_data_new(i,1)==1))
         F_Nondemented=F_Nondemented+1;
     elseif((tags_new(i)==1)&&(MF_data_new(i,1)==0))
         M_Demented=M_Demented+1;
     elseif((tags_new(i)==0)&&(MF_data_new(i,1)==0))
         M_Nondemented=M_Nondemented+1;
     end
 end
 females=0;
 for j=1:216%find how much are women
     if(MF_data_new(j,1)==1)
         females=females+1;
     end
 end
figure%The above graph indicates that men are more likely with dementia than women.
 y = [ F_Nondemented  M_Nondemented;  F_Demented   M_Demented];
 bar(y);
 set(gca,'XTickLabel',{'Nondemented','Demented'})
 legend('F','M')
 title('Gender and Demented rate')
 xlabel('Group')
 ylabel('rate')
 clr = [1 0 0;
   0 1 0];
colormap(clr);



%MMSE


GROUP_MMSE= [tags_new(:,1),cell2mat(data_new(2:end,6))];
MMSE_sort=sortrows(GROUP_MMSE,2);
Nondemented=zeros(31,2);
Score_MMSE=(0:30)';
Nondemented(:,1)=Score_MMSE;
Demented=zeros(31,2);
Demented(:,1)=Score_MMSE;


for i=1:1:216
    for j=1:1:31
        if(MMSE_sort(i,2)==Nondemented(j,1)&&MMSE_sort(i,1)==0)
            Nondemented(j,2)=Nondemented(j,2)+1;
        end
        if(MMSE_sort(i,2)==Demented(j,1)&&MMSE_sort(i,1)==1)
            Demented(j,2)=Demented(j,2)+1;
        end
    end
end
figure%The chart shows Nondemented group got much more higher MMSE scores than Demented group.
hold on
plot( Nondemented(:,1), Nondemented(:,2),'b')
plot( Demented(:,1), Demented(:,2),'g')
xlabel('MMSE Score');
ylabel('number');
title('MMSE level')
legend('Nondemented','Demented')
hold off

%nWBV


GROUP_nWBV= [tags_new(:,1),cell2mat(data_new(2:end,9))];
nWBV_sort=sortrows(GROUP_nWBV,2);
Nondemented=zeros(7,2);
Score_nWBV=zeros(7,2);
Score_nWBV=[0.60,0.65,0.70,0.75,0.80,0.85,0.90]';
Nondemented(:,1)=Score_nWBV;
Demented=zeros(7,2);
Demented(:,1)=Score_nWBV;

for i=1:1:216
    for j=1:1:7
        if(nWBV_sort(i,2)<=Nondemented(j,1)&&nWBV_sort(i,2)>Nondemented(j-1,1)&&nWBV_sort(i,1)==0)
            Nondemented(j,2)=Nondemented(j,2)+1;
        end
        if(nWBV_sort(i,2)<=Demented(j,1)&&nWBV_sort(i,2)>Demented(j-1,1)&&nWBV_sort(i,1)==1)
            Demented(j,2)=Demented(j,2)+1;
        end
    end
end
figure%The chart indicates that Nondemented group has higher brain volume ratio than Demented group. This is assumed to be because the diseases affect the brain to be shrinking its tissue.
hold on
plot( Nondemented(:,1), Nondemented(:,2),'b')
plot( Demented(:,1), Demented(:,2),'g')
xlabel('nWBV level');
ylabel('number');
title('nWBV level')
legend('Nondemented','Demented')
hold off


