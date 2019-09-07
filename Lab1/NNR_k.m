function [ number ] = NNR_k( test,train,neighbours,distances )
%distance = zeros(1,size(train,1));
%big_test = repmat(test,size(train,1),1);
%distance = sqrt(sum((big_test-train(:,2:257)).^2,2));

%parfor i=1:size(train,1)
%    distance(i) = sqrt(sum((test-train(i,2:257)).^2));
%end
[~,indexes] = sort(distances,'ascend');
counter = zeros(1,10);
%train(indexes(1:neighbours),1)+1
for j=1:neighbours
    counter(train(indexes(j),1)+1) = counter(train(indexes(j),1)+1)+1;
end
[~,number] = max(counter);
number = number-1;
end

