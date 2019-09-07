function [distance,label] = min_dist_label( test,train)

distance = zeros(1,size(train,1));
big_test = repmat(test,size(train,1),1);
distance = sqrt(sum((big_test-train(:,2:257)).^2,2));
%size(distance);
%distance = zeros(1,size(train,1));
%parfor i=1:size(train,1)
%    distance(i) = sqrt(sum((test-train(i,2:257)).^2));
%end
[~,label] = min(distance);
end

