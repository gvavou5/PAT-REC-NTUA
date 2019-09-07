clear all; close all; clc;
tic;
if ((exist('lab','dir'))==0)
    mkdir lab;
end
digitnames = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'};
%add the necessary toolboxes for this lab
addpath ./HMM 
addpath ./KPMtools 
addpath ./KPMstats 
addpath ./netlab 
addpath ./prtools
%load the C coefficients from the preparation of the lab
load('C_Coeff.mat');
speakers = 15;
trainers = 11; 
testers  =  4; %1-11 for training 12-15 for testing approximately 70% - 30%
C_coeff_final =  cellfun(@transpose,C_coeff_final,'UniformOutput',false);
C_coeff_train =  C_coeff_final(:,1:trainers);
C_coeff_test = C_coeff_final(:,speakers-testers+1:end);
%C_coeff_test = C_coeff_final(:,trainers+1:end);

%%Step 10
Nc = 13;
Ns = 5;
Nm = 2;
prior = cell(1,9);
transmat = cell(1,9);
transbins = zeros(Ns);
for i=1:Ns
    for j=1:Ns
        if ~(j<i || j>i+1)
            transbins(i,j) = 1;
        end
    end
end

%%Step 11
%help on how to use the HMM toolbox found here: https://www.cs.ubc.ca/~murphyk/Software/HMM/hmm_usage.html
for digit = 1:9
   %Nm gaussians  , Ns states , using k-means
   prior{digit} = [1 ; zeros(Ns-1,1)];
   transmat{digit} = rand(Ns, Ns);
   transmat{digit} = transmat{digit}.*transbins; 
   transmat{digit} = mk_stochastic(transmat{digit});
   %transmat{digit} = transmat{digit}./(repmat(sum(transmat{digit},2),[1,Ns]));
   [mu{digit}, Sigma{digit},weights{digit}] = mixgauss_init(Ns*Nm, [C_coeff_train{digit,:}], 'full','kmeans');
   mu{digit} = reshape(mu{digit}, [Nc Ns Nm]);
   Sigma{digit} = reshape(Sigma{digit}, [Nc Nc Ns Nm]);
   mixmat{digit} = mk_stochastic(rand(Ns, Nm));

   data =  C_coeff_train(digit,(~cellfun('isempty',C_coeff_train(digit,:))) );    
  
   %train my model
   [LL{digit}, prior_tr{digit}, transmat_tr{digit}, mu_tr{digit}, Sigma_tr{digit}, mixmat_tr{digit}] = ...
    mhmm_em(data, prior{digit}, transmat{digit}, mu{digit}, Sigma{digit}, mixmat{digit}, 'max_iter', 15,'verbose', 0);
    
   [LL_iter{digit},~ , ~ , ~ , ~ , ~] = ...
    mhmm_em_no_convergence(data, prior{digit}, transmat{digit}, mu{digit}, Sigma{digit}, mixmat{digit}, 'max_iter', 15,'verbose', 0);
       
end

%%Step 12 
loglik = zeros(9,1);
%errors = zeros(9,1);
results = zeros(9,speakers-trainers);
for i=1:9
    for j=1:testers
        data = C_coeff_test(i,j);
        if isempty(data{1})
            continue;
        end
        %restoredefaultpath
        %rehash toolboxcache
        %savepath
        %addpath ./HMM 
        %addpath ./KPMtools  
        %addpath ./KPMstats 
        %addpath ./netlab 
        %addpath ./prtools
        for digit = 1:9
            loglik(digit) = mhmm_logprob(data, prior_tr{digit}, transmat_tr{digit}, mu_tr{digit}, Sigma_tr{digit}, mixmat_tr{digit});
        end
        [~,index] = max(loglik);
        results(i,j) = index;
    end
end

%%Step 13
k1 = 3 ; %Vavouliotis Georgios 03112083
%if stopped at convergence
figure();
plot(LL{k1});
title('LogProb For k=3 stopped at convergence')
print -djpeg '.\lab\LogProb_For_k=3_stopped_at_convergence.jpg'
%if forced to do N iterations
figure();
plot(LL_iter{k1});
title('LogProb For k=3 with 15 forced iterations')
print -djpeg '.\lab\LogProb_For_k=3_with_15_forced_iterations.jpg'

k1 = 7 ; %Stavrakakis Dimitrios 03112017
%if stopped at convergence
figure();
plot(LL{k1});
title('LogProb For k=7 stopped at convergence')
print -djpeg '.\lab\LogProb_For_k=7_stopped_at_convergence.jpg'
%if forced to do N iterations
figure();
plot(LL_iter{k1});
title('LogProb For k=7 with 15 forced_iterations')
print -djpeg '.\lab\LogProb_For_k=7_with_15_forced_iterations.jpg'

%%Step 14
Confusion = zeros(9);

for i = 1:size(results,1)
    for j = 1:size(results,2)
        if (results(i,j) ~= 0) %because one record is missing from our test data
            Confusion(i,results(i,j)) = Confusion(i,results(i,j)) + 1;
        end
    end
end

total_test_data = numel(C_coeff_test(~cellfun('isempty',C_coeff_test)));
successful_classifications = trace(Confusion);
total_success_ratio = successful_classifications / total_test_data * 100

%%Step 15
%generate differenct colours for our viterbi path plots of our speakers
cc = hsv(speakers-trainers+2);
for digit=1:9
   filename = sprintf('.\\lab\\Viterbi_path_for_digit=%d.jpg',digit);
   figure('Name','Viterbi Path','NumberTitle','off');
   for j=1:testers%trainers%speakers-trainers
        data = C_coeff_test(i,j);
        if isempty(data{1})
            continue;
        end
        %Computing the most probable sequence (Viterbi)
        %First you need to evaluate B(t,i) = P(y_t | Q_t=i) for all t,i:
        B = mixgauss_prob(data{1}, mu_tr{digit}, Sigma_tr{digit}, mixmat_tr{digit}); 
        [path] = viterbi_path(prior_tr{digit}, transmat_tr{digit}, B);
        plot(path,'color', cc(j,:),'linestyle','--','marker','*');
        hold on;
        grid on;
   end
   title(['Viterbi Path for digit: ',num2str(digit)]);
   print(filename,'-djpeg')
   hold off;
end

toc;
