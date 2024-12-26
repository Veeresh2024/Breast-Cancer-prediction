function [Tag_new,tag_test_without_NAN]=Decision_tree_without_NAN(set_without_NAN,set_tags_without_NAN)
%for without_NAN 70,30:
train_without_NAN=[];
tag_train_without_NAN=[];
test_without_NAN=[];
tag_test_without_NAN=[];

idx = randperm(354);%generate random vector of size 371
for j=1:247%training
    train_without_NAN=[train_without_NAN; set_without_NAN(idx(j),:)];
    tag_train_without_NAN=[tag_train_without_NAN;set_tags_without_NAN(idx(j),:)];
           
end


for j=248:1:354% test
    test_without_NAN=[test_without_NAN; set_without_NAN(idx(j),:)];
    tag_test_without_NAN=[tag_test_without_NAN;set_tags_without_NAN(idx(j),:)];
       
end

Mdl = fitctree(train_without_NAN,tag_train_without_NAN);
view(Mdl,'mode','graph');
Tag_new = predict(Mdl,test_without_NAN);
sum=0;
for i=1:107
    if(strcmp(Tag_new(i),tag_test_without_NAN(i)))
        sum=sum+1;
    end
end

answer=['without_NAN: The decision tree predicts right ',num2str(sum),' from test set of 107 samples'];
disp(answer);
cvrtree = crossval(Mdl);
cvloss= kfoldLoss(cvrtree);
resuberror=resubLoss(Mdl);
x=['The error rate of the cv is: ',num2str(cvloss), ' The error rate of the training is ',num2str(resuberror)];
disp(x);
%{
%Prune the tree until pruning parameter = 4 and evaluate the performances 

for i=0:1:4
    Mdl = fitctree(train_without_NAN,tag_train_without_NAN);
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
resuberror=zeros(3,1);
err = zeros(3,1);
prune_level=zeros(3,1);
for n=1:3
    prune_level(n)=n;
    Mdl = fitctree(train_without_NAN,tag_train_without_NAN);
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
title('performance as a function of pruning (without_NAN)')
legend('training','cv sets')

tot_sum_without_NAN=sum;

end