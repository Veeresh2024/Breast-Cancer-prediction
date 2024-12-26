function[results_new_decision,Tag_new,tag_test_data]=Decision_tree_data(set_data,set_tags_data,set_data_new)
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

Mdl = fitctree(train_data,tag_train_data);
view(Mdl,'mode','graph');
Tag_new = predict(Mdl,test_data);
results_new_decision=predict(Mdl,set_data_new);
sum=0;
for i=1:111
    if(strcmp(Tag_new(i),tag_test_data(i)))
        sum=sum+1;
    end
end

answer=['Date: The decision tree predicts right ',num2str(sum),' from test set of 111 samples'];
disp(answer);
cvrtree = crossval(Mdl);
cvloss= kfoldLoss(cvrtree);
resuberror=resubLoss(Mdl);
x=['The error rate of the cv is: ',num2str(cvloss), ' The error rate of the training is ',num2str(resuberror)];
disp(x);

%Prune the tree until pruning parameter = 4 and evaluate the performances 
%{
for i=0:1:4
    Mdl = fitctree(train_data,tag_train_data);
    tree = prune(Mdl,'Level',i);% pruning levels 0,1,2,3,4
    view(tree,'mode','graph')
    cvrtree2 = crossval(tree);
    cvloss2 = kfoldLoss(cvrtree2);
    resuberror2=resubLoss(tree);
    x=['The error rate of the cv in prunning size ',num2str(i),' is: ',num2str(cvloss2),' The error rate of the training is ',num2str(resuberror2)];
    disp(x);
    
end
%}

%Ploting a graph of performance as a function of pruning, both for the training and the CV sets (in one graph)
resuberror=zeros(4,1);
err = zeros(4,1);
prune_level=zeros(4,1);
for n=1:4
    prune_level(n)=n;
    Mdl = fitctree(train_data,tag_train_data);
    t  = prune(Mdl,'Level',n);
    cvrtree3 = crossval(t);
    err(n) = kfoldLoss(cvrtree3);
   resuberror(n)=resubLoss(t);
    
end

figure
hold on
plot(prune_level, resuberror,'b');
plot(prune_level, err,'r');
xlabel('pruning level');
ylabel('error');
title('performance as a function of pruning (Data)')
legend('training','cv sets')

tot_sum_data=sum;

end