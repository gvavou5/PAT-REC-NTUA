function [classifications, successRate] = nnr( traindata, testdata, k)
    for i = 1:size(testdata, 1)
        testchars = testdata(i,1:166);
        testchars = repmat(testchars, size(traindata,1),1);
        distAll(i, :) =sqrt(sum((testchars - traindata(:, 1:166)).^2,2))';
    end
    [~, ind] = sort(distAll,2);
    if k == 1
        classifications = traindata(ind(:,1), 167);
    else
        tmptraindata = traindata(:,167);
        [classifications, ~] = mode(tmptraindata(ind(:, 1:k)),2);        
    end
    successRate = sum(classifications == testdata(:, 167))/size(testdata,1);
end

