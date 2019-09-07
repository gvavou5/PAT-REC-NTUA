function [ output_cell ] = Find_mean_from_data( input_cell,Nc )
counters = cell(size(input_cell,1),1);
for i=1:size(input_cell,1)
        counters{i} = 0;
        output_cell{i} = zeros(1,Nc);
        for j=1:size(input_cell,2)
            if (~isnan(input_cell{i,j}))
                counters{i} = counters{i} + 1;
                output_cell{i} = output_cell{i} + input_cell{i,j} ;
            end
        end               
end
output_cell = output_cell';
output_cell = cellfun(@(x,y)x/y,output_cell,counters,'UniformOutput', false);
end

