%% Lab 2 Pattern Recognition
close all;


addpath ./HMM ./KPMtools ./KPMstats ./Netlab ./prtools
load('c.mat');
C_keep_only13=Ctemp;
% C_keep_only13 = cellfun(@transpose,C_keep_only13,'UniformOutput',false);

%% Step 11
Nm = 1; % # gaussians
Ns = 5; % # states
Nc=13;%num of coefficients for each signal <=> features
% prior = cell(1,9); prior_trained = cell(9,1);
% transitions = cell(9,1); transitions_trained = cell(9,1);
% mean_val = cell(9,1); mean_val_trained = cell(9,1);
% variance = cell(9,1); variance_trained = cell(9,1);
%  mixmat_trained = cell(9,1);
% LL = cell(9,1);
train_ratio=11;
%70%*((7*15+2*14)/9)=10,34->10 or 11(i use 11 for better results)
for digit = 1:9 % for every digit create a hmm
    
    % initial prior probability
    %0-> i<>1
    %1-> i=1
    prior{digit} = zeros(Ns,1);
    prior{digit}(1) = 1;

    
    % left-right hmm1 transition matrix
    % with 
    transitions{digit} = rand(Ns,Ns);
    for i = 1:Ns
        for j = 1:Ns
            if (i>j)
                transitions{digit}(i,j) = 0; 
            end
            if (i+1<j)
                transitions{digit}(i,j) = 0; 
            end
        end
    end
    transitions{digit} = mk_stochastic(transitions{digit});
  %  transitions=transitions';
%      for h=1:15
%          C_keep_only13{digit,h}=C_keep_only13{digit,h}';
%      end
    % initialize mixture of gauusians    
    % mixture of gauss (kmeans is not needed is set by default)
    [mean_val{digit}, variance{digit}] = mixgauss_init(Ns*Nm, [C_keep_only13{digit,1:train_ratio}], 'full');
    %mean_val=mean_val';
    mean_val{digit} = reshape(mean_val{digit}, [Nc Ns Nm]);
    %variance=variance';
    variance{digit} = reshape(variance{digit}, [Nc Nc Ns Nm]);
    mixmat{digit} = mk_stochastic(rand(Ns,Nm));
   % mixmat=mixmat';
     % fix data for omitted speakers
    if(digit == 8) 
        mhmm_em_data = C_keep_only13(digit,[1:6 8:train_ratio]);
    else    
        mhmm_em_data = C_keep_only13(digit,1:train_ratio);
    end
    %% Step 12
    % improve estimates using EM
    [LL{digit}, prior_trained{digit}, transitions_trained{digit}, mean_val_trained{digit},...
    variance_trained{digit}, mixmat_trained{digit}] = mhmm_em(mhmm_em_data, prior{digit},...
    transitions{digit}, mean_val{digit}, variance{digit}, mixmat{digit},'max_iter', 10,'verbose', 0);
    
end
k1=mod(03112074,10);
figure('Name','Learning Curve for digit: 4','NumberTitle','off');
plot(LL{k1},'*r--');
grid on; title('Learning Curve for digit: 4');
xlabel('#Iterations');
ylabel('LogLikelihood');

B=cell(9,4);
for i=1:9
     for j=(train_ratio+1):15
       B{i,j-11}=C_keep_only13{i,j};
     end
end

for i=1:9
    for j=(train_ratio+1):15
        if ((i==6)&&(j==12)) 
            continue;
        end
%           restoredefaultpath
%           rehash toolboxcache
%           savepath
%          digitaudioname = sprintf('./digits2016/%s%d.wav', strjoin(digitnames(i)), j);
%           C{i,j-11} = extractCharacteristics(digitaudioname, 0.025,0.01,13);
%           addpath ./HMM ./KPMtools ./KPMstats ./Netlab ./prtools

        for digit = 1:9
            likelihood(digit) = mhmm_logprob(B{i,j-11}, prior_trained{digit}, transitions_trained{digit},...
                mean_val_trained{digit}, variance_trained{digit}, mixmat_trained{digit});
        end
        
        [~, hmm_digit_classifier(i,j-11)] = max(likelihood); 
    end
end


ConfusionMatrix = zeros(9,9);
for i=1:9
    for j=1:9
        
        ConfusionMatrix(i,j) = size(find(hmm_digit_classifier(i,:)==j),2);
    end
end

figure('Name','Confusion Matrix','NumberTitle','off');
speakers = 4;
image(ConfusionMatrix);
colormap(bone(speakers));
xlabel('#Classified Utterances');
ylabel('Digit Utterances');

SuccessRate = trace(ConfusionMatrix)/sum(sum(ConfusionMatrix));
text(0,0,['SuccessRate = ',num2str(SuccessRate*100),'%']);
