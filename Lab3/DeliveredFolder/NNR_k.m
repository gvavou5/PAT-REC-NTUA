function [ results ] = NNR_k( test,train,neighbours )
%calculate the distances for test data from all the train data
temptest = test(:,1:(end-1)); %we simply remove the classification label
temptrain = train(:,1:(end-1)); 
dist = zeros(size(test,1),size(train,1));
for i=1:size(test,1)
    temp = repmat(temptest(i,:),size(train,1),1);
    dist(i,:,:) = sqrt(sum((temp-temptrain).^2,2));    
end
%sort the distances along the 2nd dimension
[~,indexes] = sort(dist,2,'ascend');

%find the classification judging by the k nearest neighbours
indexes = indexes(:,1:neighbours);

results = zeros(size(test,1),1);

for i=1:size(test,1)
    results(i) = sum(train(indexes(i,:),end));
end

results(results>+1) = +1;
results(results<-1) = -1;
end

