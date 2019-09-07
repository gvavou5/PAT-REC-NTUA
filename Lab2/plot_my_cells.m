function [out] = plot_my_cells( input_array,n1,n2,marker_specifier,color,number)
%function to plot step 9 results
if (~isnan(input_array))
        out=plot(input_array(n1),input_array(n2),marker_specifier);%[marker_specifier,color]);
        set(out(1),'color',color)
        if (nargin == 6)
            set(out,'linewidth',2);
            mylegend(out,{number},'Orientation','vertical');
        end
        grid on; hold on;
        return;
end
out = NaN;
end

