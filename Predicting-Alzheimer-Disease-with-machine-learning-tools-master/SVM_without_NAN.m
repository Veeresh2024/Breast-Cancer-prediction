function [indx_Demented,indx_Nondemented,label_Demented,label_Nondemented]=SVM_without_NAN(set_without_NAN,set_tags_without_NAN)

%for without_NAN 70,30:
train_without_NAN=[];
tag_train_without_NAN=[];
test_without_NAN=[];
tag_test_without_NAN=[];

idx = randperm(354);%generate random vector of size 371
for j=1:1:247%training
    train_without_NAN=[train_without_NAN; set_without_NAN(idx(j),:)];
    tag_train_without_NAN=[tag_train_without_NAN;set_tags_without_NAN(idx(j),:)];
           
end




for j=248:1:354% test
    test_without_NAN=[test_without_NAN; set_without_NAN(idx(j),:)];
    tag_test_without_NAN=[tag_test_without_NAN;set_tags_without_NAN(idx(j),:)];
        
        
end


%corrplot(set_data)

% svm model for 3 classes
SVMModels = cell(2,1);

classes = unique(tag_train_without_NAN(~cellfun(@isempty, tag_train_without_NAN)));
rng(1); % For reproducibility
kernel={'rbf','gaussian','linear'};
for t=1:3
    for j = 1:numel(classes)
        indx = strcmp(tag_train_without_NAN,classes(j)); % Create binary classes for each classifier
        SVMModels{j} = fitcsvm(train_without_NAN,indx,'ClassNames',[false true],'Standardize',true,'KernelFunction',kernel{t},'BoxConstraint',1);
    end
    
    indx_Demented= strcmp(tag_test_without_NAN,classes(1));
    [label_Demented,score_Demented] = predict(SVMModels{1},test_without_NAN);
    indx_Nondemented= strcmp(tag_test_without_NAN,classes(2));
    [label_Nondemented,score_Nondemented] = predict(SVMModels{2},test_without_NAN);
    sum=0;
    for i=1:107
        
        if(indx_Demented(i)==1&&label_Demented(i)==1)
            sum=sum+1;
        end
        if(indx_Nondemented(i)==1&&label_Nondemented(i)==1)
            sum=sum+1;
        end
    end
      answer=['without_NAN SVM:'];
      disp(answer);
    if (t==1)
        answer=['rbf predicts right ',num2str(sum),' from test set of 107 samples'];
        disp(answer);
    elseif (t==2)
        answer=['gaussian predicts right ',num2str(sum),' from test set of 107 samples'];
        disp(answer);
    elseif (t==3)
        answer=['linear predicts right ',num2str(sum),' from test set of 107 samples'];
        disp(answer);
    end
end
tot_sum_without_NAN=sum;


end