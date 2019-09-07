clc;clear all;close all;
%%Step 10
%load('data.mat');
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

%set 1 : features from step 6
feat_set_1 = (cell2mat(MIRFeatures));
feat_set_1_act = [feat_set_1(:,1:10) mean_Act_quant'];
feat_set_1_val = [feat_set_1(:,1:10) mean_Val_quant'];
%set 2 : features from step 7
feat_set_2 = (cell2mat(MIRFeatures));
feat_set_2 = feat_set_2(:,11:end);
feat_set_2_act = [feat_set_2(:,:) mean_Act_quant'];
feat_set_2_val = [feat_set_2(:,:) mean_Val_quant'];
%set 3 : features from step 6 and 7 combined
feat_set_3 = (cell2mat(MIRFeatures));
feat_set_3_act = [feat_set_3(:,:) mean_Act_quant'];
feat_set_3_val = [feat_set_3(:,:) mean_Val_quant'];

%delete all the ignored samples from step 10 from all features' sets
%activation
feat_set_1_act = feat_set_1_act(mean_Act_quant~=0,:);
feat_set_2_act = feat_set_2_act(mean_Act_quant~=0,:);
feat_set_3_act = feat_set_3_act(mean_Act_quant~=0,:);
%valence
feat_set_1_val = feat_set_1_val(mean_Val_quant~=0,:);
feat_set_2_val = feat_set_2_val(mean_Val_quant~=0,:);
feat_set_3_val = feat_set_3_val(mean_Val_quant~=0,:);


number_of_folds = 3;
max_number_of_neighbours = 7;
NNR_Evaluation_act_1 = zeros(number_of_folds,ceil(max_number_of_neighbours/2),7);
NNR_Evaluation_act_2 = zeros(number_of_folds,ceil(max_number_of_neighbours/2),7);
NNR_Evaluation_act_3 = zeros(number_of_folds,ceil(max_number_of_neighbours/2),7);
NNR_Evaluation_val_1 = zeros(number_of_folds,ceil(max_number_of_neighbours/2),7);
NNR_Evaluation_val_2 = zeros(number_of_folds,ceil(max_number_of_neighbours/2),7);
NNR_Evaluation_val_3 = zeros(number_of_folds,ceil(max_number_of_neighbours/2),7);

BAYES_Evaluation_act_1 = zeros(number_of_folds,7);
BAYES_Evaluation_act_2 = zeros(number_of_folds,7);
BAYES_Evaluation_act_3 = zeros(number_of_folds,7);
BAYES_Evaluation_val_1 = zeros(number_of_folds,7);
BAYES_Evaluation_val_2 = zeros(number_of_folds,7);
BAYES_Evaluation_val_3 = zeros(number_of_folds,7);


%loop for 3(number_of_folds) fold cross validation 
for fold=1:number_of_folds 
    
    %take 80% training set and 20% validation set randomly
    %activation random indexes
    act_indexes = randperm(size(feat_set_1_act,1));  
    act_indexes = act_indexes';
    feat_set_1_act = feat_set_1_act(act_indexes,:);
    feat_set_2_act = feat_set_2_act(act_indexes,:);
    feat_set_3_act = feat_set_3_act(act_indexes,:);
    %valence random indexes
    val_indexes = randperm(size(feat_set_1_val,1));    
    val_indexes = val_indexes';
    feat_set_1_val = feat_set_1_val(val_indexes,:);
    feat_set_2_val = feat_set_2_val(val_indexes,:);
    feat_set_3_val = feat_set_3_val(val_indexes,:);
    
    %above we have the whole set of our features
    %now we will divide these set into training and validation sets
    training_set_size_act = ceil(0.8*size(feat_set_1_act,1));
    training_set_size_val = ceil(0.8*size(feat_set_1_val,1));
    %set1
    feat_set_1_act_train = feat_set_1_act(1:training_set_size_act,:);
    feat_set_1_act_valid = feat_set_1_act(training_set_size_act+1:end,:);
    feat_set_1_val_train = feat_set_1_val(1:training_set_size_val,:);
    feat_set_1_val_valid = feat_set_1_val(training_set_size_val+1:end,:);
    %set2
    feat_set_2_act_train = feat_set_2_act(1:training_set_size_act,:);
    feat_set_2_act_valid = feat_set_2_act(training_set_size_act+1:end,:);
    feat_set_2_val_train = feat_set_2_val(1:training_set_size_val,:);
    feat_set_2_val_valid = feat_set_2_val(training_set_size_val+1:end,:);
    %set3
    feat_set_3_act_train = feat_set_3_act(1:training_set_size_act,:);
    feat_set_3_act_valid = feat_set_3_act(training_set_size_act+1:end,:);
    feat_set_3_val_train = feat_set_3_val(1:training_set_size_val,:);
    feat_set_3_val_valid = feat_set_3_val(training_set_size_val+1:end,:);
    
   
    %% Step 12 - NNR
    for k = 1:2:max_number_of_neighbours %test for different number of neighbours
        %Evaluate returns 6 doubles: accuracy sensitivity specificity precision recall f_measure gmean
        % Activation
        nnr_set_1_act_res = NNR_k(feat_set_1_act_valid,feat_set_1_act_train,k);
        NNR_Evaluation_act_1(fold,ceil(k/2),:)= Evaluate(feat_set_1_act_valid(:,end),nnr_set_1_act_res);

        nnr_set_2_act_res = NNR_k(feat_set_2_act_valid,feat_set_2_act_train,k);
        NNR_Evaluation_act_2(fold,ceil(k/2),:) = Evaluate(feat_set_2_act_valid(:,end),nnr_set_2_act_res);

        nnr_set_3_act_res = NNR_k(feat_set_3_act_valid,feat_set_3_act_train,k);
        NNR_Evaluation_act_3(fold,ceil(k/2),:) = Evaluate(feat_set_3_act_valid(:,end),nnr_set_3_act_res);

        % Valence
        nnr_set_1_val_res = NNR_k(feat_set_1_val_valid,feat_set_1_val_train,k);
        NNR_Evaluation_val_1(fold,ceil(k/2),:) = Evaluate(feat_set_1_val_valid(:,end),nnr_set_1_val_res);

        nnr_set_2_val_res = NNR_k(feat_set_2_val_valid,feat_set_2_val_train,k);
        NNR_Evaluation_val_2(fold,ceil(k/2),:) = Evaluate(feat_set_2_val_valid(:,end),nnr_set_2_val_res);

        nnr_set_3_val_res = NNR_k(feat_set_3_val_valid,feat_set_3_val_train,k);
        NNR_Evaluation_val_3(fold,ceil(k/2),:) = Evaluate(feat_set_3_val_valid(:,end),nnr_set_3_val_res);
    end  
    
    %%Step 13 - Naive Bayes
    bayes_set_1_act_res = BAYES(feat_set_1_act_valid,feat_set_1_act_train);
    BAYES_Evaluation_act_1(fold,:) = Evaluate(feat_set_1_act_valid(:,end),bayes_set_1_act_res);
    
    bayes_set_2_act_res = BAYES(feat_set_2_act_valid,feat_set_2_act_train);
    BAYES_Evaluation_act_2(fold,:) = Evaluate(feat_set_2_act_valid(:,end),bayes_set_2_act_res);
    
    bayes_set_3_act_res = BAYES(feat_set_3_act_valid,feat_set_3_act_train);
    BAYES_Evaluation_act_3(fold,:) = Evaluate(feat_set_3_act_valid(:,end),bayes_set_3_act_res);
    
    bayes_set_1_val_res = BAYES(feat_set_1_val_valid,feat_set_1_val_train);
    BAYES_Evaluation_val_1(fold,:) = Evaluate(feat_set_1_val_valid(:,end),bayes_set_1_val_res);
    
    bayes_set_2_val_res = BAYES(feat_set_2_val_valid,feat_set_2_val_train);
    BAYES_Evaluation_val_2(fold,:) = Evaluate(feat_set_2_val_valid(:,end),bayes_set_2_val_res);
    
    bayes_set_3_val_res = BAYES(feat_set_3_val_valid,feat_set_3_val_train);
    BAYES_Evaluation_val_3(fold,:) = Evaluate(feat_set_3_val_valid(:,end),bayes_set_3_val_res);
      
end

%Calculate mean values for our above two methods
%Set 1 Activation
mean_NNR_set_1_act = squeeze(mean(NNR_Evaluation_act_1,1));
mean_BAYES_set_1_act = mean(BAYES_Evaluation_act_1,1);
%Set 1 Valence
mean_NNR_set_1_val = squeeze(mean(NNR_Evaluation_val_1,1));
mean_BAYES_set_1_val = mean(BAYES_Evaluation_val_1,1);
%Set 2 Activation
mean_NNR_set_2_act = squeeze(mean(NNR_Evaluation_act_2,1));
mean_BAYES_set_2_act = mean(BAYES_Evaluation_act_2,1);
%Set 2 Valence
mean_NNR_set_2_val = squeeze(mean(NNR_Evaluation_val_2,1));
mean_BAYES_set_2_val = mean(BAYES_Evaluation_val_2,1);
%Set 3 Activation
mean_NNR_set_3_act = squeeze(mean(NNR_Evaluation_act_3,1));
mean_BAYES_set_3_act = mean(BAYES_Evaluation_act_3,1);
%Set 3 Valence
mean_NNR_set_3_val = squeeze(mean(NNR_Evaluation_val_3,1));
mean_BAYES_set_3_val = mean(BAYES_Evaluation_val_3,1);



%%Step 15 Expertising with WEKA
%save the necessary files

addpath(genpath('C:\Users\dimst\Desktop\Pattern_Recognition\Lab3\wekaToolbox\'));
addpath(genpath('C:\program files\Weka-3-7\'));
javaaddpath('C:\program files\Weka-3-7\weka.jar')
mkdir('./arff_files/');
%save .arff files for Set 1
labels1 = cell(size(feat_set_1_act,2),1);
for i = 1:size(feat_set_1_act,2)-1
    labels1{i} = ['feature',num2str(i)];
end
labels1{end} = 'Activation';
WekaInstance = matlab2weka('Act_Set_1',labels1,feat_set_1_act);
saveARFF('./arff_files/act_set_1.arff',WekaInstance);
labels1{end} = 'Valence';
WekaInstance = matlab2weka('Val_Set_1',labels1,feat_set_1_val);
saveARFF('./arff_files/val_set_1.arff',WekaInstance);

%save .arff files for Set 2
labels2 = cell(size(feat_set_2_act,2),1);
for i = 1:size(feat_set_2_act,2)-1
    labels2{i} = ['feature',num2str(i)];
end
labels2{end} = 'Activation';
WekaInstance = matlab2weka('Act_Set_2',labels2,feat_set_2_act);
saveARFF('./arff_files/act_set_2.arff',WekaInstance);
labels2{end} = 'Valence';
WekaInstance = matlab2weka('Val_Set_2',labels2,feat_set_2_val);
saveARFF('./arff_files/val_set_2.arff',WekaInstance);

%save .arff files for Set 3
labels3 = cell(size(feat_set_3_act,2),1);
for i = 1:size(feat_set_3_act,2)-1
    labels3{i} = ['feature',num2str(i)];
end
labels3{end} = 'Activation';
WekaInstance = matlab2weka('Act_Set_1',labels3,feat_set_3_act);
saveARFF('./arff_files/act_set_3.arff',WekaInstance);
labels3{end} = 'Valence';
WekaInstance = matlab2weka('Val_Set_1',labels3,feat_set_3_val);
saveARFF('./arff_files/val_set_3.arff',WekaInstance);











