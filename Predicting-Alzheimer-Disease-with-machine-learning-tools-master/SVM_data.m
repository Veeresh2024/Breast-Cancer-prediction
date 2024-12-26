function [indx_Demented,indx_Nondemented,label_Demented,label_Nondemented]=SVM_data(set_data,set_tags_data)
%for data 70,30:
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


%corrplot(set_data)

% svm model for 3 classes
SVMModels = cell(3,1);

classes = unique(tag_train_data(~cellfun(@isempty, tag_train_data)));
rng(1); % For reproducibility
kernel={'rbf','gaussian','linear'};
for t=1:3
    for j = 1:numel(classes)
        indx = strcmp(tag_train_data,classes(j)); % Create binary classes for each classifier
        SVMModels{j} = fitcsvm(train_data,indx,'ClassNames',[false true],'Standardize',true,'KernelFunction',kernel{t},'BoxConstraint',1);
    end
    indx_Demented= strcmp(tag_test_data,classes(1));
    [label_Demented,score_Demented] = predict(SVMModels{1},test_data);
    indx_Nondemented= strcmp(tag_test_data,classes(2));
    [label_Nondemented,score_Nondemented] = predict(SVMModels{2},test_data);
    sum=0;
    for i=1:111
        if(indx_Demented(i)==1&&label_Demented(i)==1)
            sum=sum+1;
        end
        if(indx_Nondemented(i)==1&&label_Nondemented(i)==1)
            sum=sum+1;
        end
    end
      answer=['Data SVM:'];
      disp(answer);
    if (t==1)
        answer=['rbf predicts right ',num2str(sum),' from test set of 111 samples'];
        disp(answer);
    elseif (t==2)
        answer=['gaussian predicts right ',num2str(sum),' from test set of 111 samples'];
        disp(answer);
    elseif (t==3)
        answer=['linear predicts right ',num2str(sum),' from test set of 111 samples'];
        disp(answer);
    end
end
tot_sum_data=sum;

end