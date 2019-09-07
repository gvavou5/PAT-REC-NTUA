function [ results ] = BAYES( test,train )
%bayesian classificator

%split train data into 2 matrices
%+1 Label vs -1 Label
train_p = train(train(:,end)==1,:);
train_n = train(train(:,end)==-1,:);

temptest = test(:,1:(end-1)); %we simply remove the classification label
temptrain_p = train_p(:,1:(end-1));
temptrain_n = train_n(:,1:(end-1));

%calculate mean value and standard deviation for both positives' and
%negatives' class
mean_p = sum(temptrain_p) / size(temptrain_p,1);
mean_n = sum(temptrain_n) / size(temptrain_n,1);
std_p = sqrt(sum((temptrain_p-repmat(mean_p,size(temptrain_p,1),1)).^2)/size(temptrain_p,1)); 
std_n = sqrt(sum((temptrain_n-repmat(mean_n,size(temptrain_n,1),1)).^2)/size(temptrain_n,1));

apriori    = zeros(2,1);   % array of a-priori probabilities
apriori(1) = size(temptrain_p,1)/size(train,1);
apriori(2) = size(temptrain_n,1)/size(train,1);
mean_value = [mean_p ; mean_n];
std_value = [std_p ; std_n];

results = zeros(size(test,1),1);
aposteriori = zeros(size(test,1),2);

for i = 1:size(test,1)
    for j = 1:2
       gauss(j,:)  = ((1/sqrt(2*pi))*(std_value(j,:).^(-1))).*(exp(-0.5*((temptest(i,1:end)-mean_value(j,:)).^2)./(std_value(j,:))));
       aposteriori(i,j)= sum(log(gauss(j,:))) + log(apriori(j));
       %aposteriori = aposteriori';
    end
    
    [~,idx] = max(aposteriori(i,:));
    
    if idx == 1
        results(i) = 1;
    else
        results(i) = -1;
    end    
    
end



end

