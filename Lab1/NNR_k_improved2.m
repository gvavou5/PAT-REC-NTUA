function [ number ] = NNR_k_improved2( test,train,neighbours,distances )
%NNR_k with weights

[~,indexes] = sort(distances,'ascend');
%counter = zeros(1,10);
weights = 1./distances;
%train(indexes(1:neighbours),1)+1
weights_sums = zeros(1,10);

for j = 1:neighbours
   weights_sums(train(indexes(j),1)+1) =  weights_sums(train(indexes(j),1)+1) + weights(indexes(j));
end

[~,number] = max(weights_sums);
number = number-1;
end
