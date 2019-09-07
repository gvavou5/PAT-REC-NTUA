clc;clear all;close all;
%%Step 10
load('MIRFeatures.mat');
load('Act_Val.mat');
%apply the threshold
mean_Act_quant(mean_Act_quant < 3) = +1; %high activation
mean_Val_quant(mean_Val_quant < 3) = -1; %negative valence
mean_Act_quant(mean_Act_quant > 3) = -1; %low activation
mean_Val_quant(mean_Val_quant > 3) = +1; %positive valence
mean_Act_quant(mean_Act_quant == 3) = 0;
mean_Val_quant(mean_Val_quant == 3) = 0;

val_samples = size(mean_Val_quant(mean_Val_quant ~= 0),2);
act_samples = size(mean_Act_quant(mean_Act_quant ~= 0),2);

disp(['Final number of samples for activation: ', num2str(act_samples)]);
disp(['Final number of samples for valence: ', num2str(val_samples)]);

%%Step 11
%Features' Set Preparation


%set 3 : features from step 6 and 7 combined
feat_set_3 = (cell2mat(MIRFeatures));
feat_set_3_act = feat_set_3(mean_Act_quant~=0,:);
temp_Act_quant = mean_Act_quant(:,mean_Act_quant~=0);
feat_set_3_val = feat_set_3(mean_Val_quant~=0,:);
temp_Val_quant = mean_Val_quant(:,mean_Val_quant~=0);
%feat_set_3_act = [feat_set_3(:,:) mean_Act_quant'];
%feat_set_3_val = [feat_set_3(:,:) mean_Val_quant'];



number_of_folds = 3;
max_number_of_neighbours = 7;
NNR_Evaluation_act_3_pca = zeros(number_of_folds,ceil(max_number_of_neighbours/2),7);
NNR_Evaluation_val_3_pca = zeros(number_of_folds,ceil(max_number_of_neighbours/2),7);

BAYES_Evaluation_act_3_pca = zeros(number_of_folds,7);
BAYES_Evaluation_val_3_pca = zeros(number_of_folds,7);


%loop for 3(number_of_folds) fold cross validation 
for fold=1:number_of_folds 
    
    %take 80% training set and 20% validation set randomly
    %activation random indexes
    act_indexes = randperm(size(feat_set_3_act,1));  
    act_indexes = act_indexes';
    feat_set_3_act = feat_set_3_act(act_indexes,:);
    %valence random indexes
    val_indexes = randperm(size(feat_set_3_val,1));    
    val_indexes = val_indexes';
    feat_set_3_val = feat_set_3_val(val_indexes,:);
    
    %above we have the whole set of our features
    %now we will divide these set into training and validation sets
    training_set_size_act = ceil(0.8*size(feat_set_3_act,1));
    training_set_size_val = ceil(0.8*size(feat_set_3_val,1));
    
    %% Step 14 - PCA 
    p = 150; %set this to your preferred number of components
    [~,feat_set_3_act,~] = pca(feat_set_3_act, 'NumComponents',p);
    [~,feat_set_3_val,~] = pca(feat_set_3_val, 'NumComponents',p);
    temp_Act_quant = temp_Act_quant(:,act_indexes);
    temp_Val_quant = temp_Val_quant(:,val_indexes);
    
    %set3
    feat_set_3_act =[feat_set_3_act temp_Act_quant'];
    feat_set_3_val =[feat_set_3_val temp_Val_quant'];
    
    feat_set_3_act_train = feat_set_3_act(1:training_set_size_act,:);
    feat_set_3_act_valid = feat_set_3_act(training_set_size_act+1:end,:);
    feat_set_3_val_train = feat_set_3_val(1:training_set_size_val,:);
    feat_set_3_val_valid = feat_set_3_val(training_set_size_val+1:end,:);
    
   
    %% Step 12 - NNR
    for k = 1:2:max_number_of_neighbours %test for different number of neighbours
        %Evaluate returns 6 doubles: accuracy sensitivity specificity precision recall f_measure gmean
        % Activation
        nnr_set_3_act_res = NNR_k(feat_set_3_act_valid,feat_set_3_act_train,k);
        NNR_Evaluation_act_3_pca(fold,ceil(k/2),:) = Evaluate(feat_set_3_act_valid(:,end),nnr_set_3_act_res);

        % Valence
        nnr_set_3_val_res = NNR_k(feat_set_3_val_valid,feat_set_3_val_train,k);
        NNR_Evaluation_val_3_pca(fold,ceil(k/2),:) = Evaluate(feat_set_3_val_valid(:,end),nnr_set_3_val_res);
    end  
    
    %%Step 13 - Naive Bayes
    bayes_set_3_act_res = BAYES(feat_set_3_act_valid,feat_set_3_act_train);
    BAYES_Evaluation_act_3_pca(fold,:) = Evaluate(feat_set_3_act_valid(:,end),bayes_set_3_act_res);

    bayes_set_3_val_res = BAYES(feat_set_3_val_valid,feat_set_3_val_train);
    BAYES_Evaluation_val_3_pca(fold,:) = Evaluate(feat_set_3_val_valid(:,end),bayes_set_3_val_res);
      
end

%Calculate mean values for our above two methods

%Set 3 Activation
mean_NNR_set_3_act_pca = squeeze(mean(NNR_Evaluation_act_3_pca,1));
mean_BAYES_set_3_act_pca = mean(BAYES_Evaluation_act_3_pca,1);
%Set 3 Valence
mean_NNR_set_3_val_pca = squeeze(mean(NNR_Evaluation_val_3_pca,1));
mean_BAYES_set_3_val_pca = mean(BAYES_Evaluation_val_3_pca,1);


%save .arff files for Set 3
addpath(genpath('C:\Users\dimst\Desktop\Pattern_Recognition\Lab3\wekaToolbox\'));
addpath(genpath('C:\program files\Weka-3-7\'));
javaaddpath('C:\program files\Weka-3-7\weka.jar')
mkdir('./arff_files/');
labels3 = cell(size(feat_set_3_act,2),1);
for i = 1:size(feat_set_3_act,2)-1
    labels3{i} = ['feature',num2str(i)];
end
labels3{end} = 'Activation';
WekaInstance = matlab2weka('Act_Set_1_pca',labels3,feat_set_3_act);
saveARFF('./arff_files/act_set_3_pca.arff',WekaInstance);
labels3{end} = 'Valence';
WekaInstance = matlab2weka('Val_Set_1_pca',labels3,feat_set_3_val);
saveARFF('./arff_files/val_set_3_pca.arff',WekaInstance);
