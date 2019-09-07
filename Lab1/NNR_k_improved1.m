function [ number ] = NNR_k_improved1( test,train,neighbours,distances )
%NNR_k with mean
[~,indexes] = sort(distances,'ascend');
counter = zeros(1,10);
dist_sum = zeros(1,10);

for j=1:neighbours
    counter(train(indexes(j),1)+1) = counter(train(indexes(j),1)+1)+1;
    dist_sum(train(indexes(j),1)+1) = dist_sum(train(indexes(j),1)+1) + distances(indexes(j));
end
dist_sum = dist_sum ./ counter;
dist_sum(counter == 0 ) = +inf;
[~,number] = min(dist_sum);
number = number-1;
end

