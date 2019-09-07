% Ergasthriaki Askhsh 1 - Proparaskeuh
% Sunergates : 
%              Vavouliotis Georgios  (03112083)
%              Stavrakakis Dimitrios (03112017)
%same as patrec but with 80% of train data for training
tic;

close all; clear all; clc ;

%% Bhma 1 -  Sxediasmos tou 131ou psifiou

train    = importdata('train.txt'); % pairnw ta train data se morfh pinaka 7291X257
newtestset = [];
newtrainset = [];
for digit = 0:9
    keepdigit = train(train(:,1)==digit,:);
    indexes = randperm(size(keepdigit,1));
    keepdigit = keepdigit(indexes,:);
    validationsize = floor(size(keepdigit,1)/5);
    newtestset = [newtestset; keepdigit(1:validationsize,:)];
    newtrainset = [newtrainset; keepdigit(validationsize+1:end,:)];
end 
%% Bhma 2,3 - Mesh Timh & Diaspora xarakthristikwn gia to pixel (10,10) tou 0 sta train data

sum1 = 0;    % krataei to a8roisma gia ta (10,10)
cnt  = 0;    % metarei ta mhdenika
e2new   = 0; % krataei to a8roisma twn tetragwnwn gia ta (10,10) gia na vrw th diaspora

for i = 1:size(newtrainset,1)
    if newtrainset(i,1) == 0 % an prokeitai gia 0
        zero = newtrainset(i,2:257); % pairnw ta xaraktiristika toy
        arr  = reshape(zero,[16,16]); % ta organwnw opws kai prin se 16x16
        sum1 = sum1 + arr(10,10); 
        e2new   = e2new + arr(10,10)^2;
        cnt  = cnt +1;
    end 
end

disp('Step 2 Result:');
E = sum1/cnt;  % euresh meshs timhs
disp(E);

disp('Step 3 Result:');
V = (e2new/cnt)-E^2 ; % euresh diasporas
disp(V);

%% Bhma 4 - Euresh Meshs Timhs kai Diasporas gia ola ta pixel tou 0 gia ta train data

sum_tot = zeros(16,16); % 2d array gia ta a8roismata tou ka8e pixel
e2new   = zeros(16,16); % 2d array gia to a8roisma tetragwnwn tou ka8e pixel
cnt = 0;
for i = 1:size(newtrainset,1)
    if newtrainset(i,1) == 0 % an vrw mhdeniko kanw oti kai sto bhma 2,3 
        zero     = newtrainset(i,2:257);
        arr      = reshape(zero,[16,16]);
        sum_tot  = sum_tot + arr;
        e2new    = e2new + arr.^2;
        cnt      = cnt +1;
    end 
end

E_tot   = zeros(16,16);% 2d array gia na krathsw th mesh timh gia ka8e pixel
V_tot   = zeros(16,16); % 2d array gia na krathsw th diaspora gia ka8e pixel

disp('Step 4 Results:');

disp('Mean Value Array:')
E_tot = sum_tot/cnt;     % Mesh Timh
disp(E_tot);

disp('Variance Array:');
V_tot = (e2new/cnt)-E^2; % Diaspora
disp(V_tot);


%% Bhma 5 - Sxediasmos tou 0 me bash th mesh timh
mkdir Bhma5-6Results1;
cd Bhma5-6Results1
figure(2);
imagesc(E_tot);
title('Zero using Mean Value');
print -djpeg Zero_Mean.jpeg;
cd ../

%% Bhma 6 - Sxediasmos tou 0 me bash th diaspora
cd Bhma5-6Results1 ;
figure(3);
imagesc(V_tot);
title('Zero using Variance');
print -djpeg Zero_Var.jpeg;
cd ../

%% Bhma 7 - Ypologismos E,V gia ola ta digits me xrhsh twn train data

sum_tot_all = zeros(16,16,10); % gia ka8e digit krataei to a8roisma twn pixel
e_2_all     = zeros(16,16,10); % gia ka8e digit krataei to a8roisma twn tetragwnwn twn pixel
cnt         = zeros(10);       % krataei poses fores vrhka ka8e digit gia na vrw th mesh timh

for i = 1:size(newtrainset,1) 
    k = newtrainset(i,1)+1; % einai o deikths gia tou pou 8a kanw store(to +1 ofeiletai sto oti oi pinakes sto matlab arxizoun thn ari8misi apo to 1) 
    % akolou8w thn idia diadikasia me prin
    zero = newtrainset(i,2:257);
    arr  = reshape(zero,[16,16]);
    sum_tot_all(:,:,k)  = sum_tot_all(:,:,k) + arr;
    e_2_all(:,:,k)      = e_2_all(:,:,k) + arr.^2;
    cnt(k)  = cnt(k) +1;
end

% Sxediasmos olwn twn digits me xrhsh ths meshs timhs

% pinakes meshs timhs kai diasporas gia ka8e digit 
E_tot_all = zeros(16,16,10);
V_tot_all = zeros(16,16,10);
mkdir Bhma7Results1 ;
save_path = './Bhma7Results1/%d.jpg' ;

for k = 1:10
    E_tot_all(:,:,k) = sum_tot_all(:,:,k)/cnt(k); 
    V_tot_all(:,:,k) = (e_2_all(:,:,k)/cnt(k)) - E_tot_all(:,:,k).^2;
    fig = figure(k+20);
    imagesc(E_tot_all(:,:,k));
    print(fig, sprintf(save_path, k-1), '-djpeg');
    %figure(k+30);
    %imagesc(V_tot_all(:,:,k));
    %title('digit with V');

end


%% Bhma 8 - Taksinomisi tou 101ou digit twn test data me xrhsh eukleidias apostashs
test = importdata('test.txt'); % pairnw ta train data se morfh pinaka size(newtestset,1)X257

digit101 = test(101,2:257); % pairnw ta xaraktiristika toy 101ou psifiou

matrix = reshape(digit101, [16, 16]); % ta organwnw se ena pinaka 16x16

% emfanizw to psifio auto kai to apo8hkeuw katallhla
mkdir Bhma8Results1 ;
cd Bhma8Results1;
figure(65); 
imagesc(matrix);
title('Plot-Digit101');
print -djpeg Plot101.jpeg; 
cd ../

% euresh eukleidias apostashs gia kaue digit wste na kanoume swsta to
% classification

dist = zeros(16,16,10); % pinakas eukleidiwn apostasewn
sums = zeros(10,1);     % gia ka8e digit exei to a8roisma twn eukleidiwn apostasewn

for k = 1:10
    dist(:,:,k) = (matrix(:,:) - E_tot_all(:,:,k)).^2;
    sums(k) = sum(sum(dist(:,:,k)));
end

% briskw to elaxisto a8roisma eukleidiwn apostasewn
[~,index] = min(sums(:));
disp('Step 8 --> Digit 101 of test data result = ');
mynumber = index-1 ; % brhka to digit pou o taksinomitis mou kanei match me to 101o pshfio
disp(mynumber);

%% Bhma 9 - Taksinomisi olwn twn pshfiwn twn test data se mia apo tis 10 kathgories me xrhsh eukleidias apostashs 

res = zeros(size(newtestset,1),1); % o pinakas autos 8a periexei to apotelesma tou taksinomiti mou gia ta test data

for i = 1:size(newtestset,1)
    k = newtestset(i,1)+1;  % einai o deikths gia tou pou 8a kanw store(to +1 ofeiletai sto oti oi pinakes sto matlab arxizoun thn ari8misi apo to 1) 
    
    % akolou8w th diadikasia tou Bhmatos 8
    zero = newtestset(i,2:257);
    arr  = reshape(zero,[16,16]);
    
    dist = zeros(16,16,10);
    sums = zeros(10,1);

    for k = 1:10
        dist(:,:,k) = (arr(:,:) - E_tot_all(:,:,k)).^2;
        sums(k) = sum(sum(dist(:,:,k)));
    end

    [~,index] = min(sums(:));
    res(i) = index-1;
end

euclidean_res = res;
% Success rate

cnt1 = 0;
for i = 1:size(newtestset,1)
   if res(i)== newtestset(i,1)
       cnt1=cnt1+1;
   end
end

disp('Step 9 : Success Rate =');
Ratio = (cnt1/size(newtestset,1))*100;
disp(Ratio);


% Success rate for each digit
cnt_res = zeros(10);
cnt_dig = zeros(10);

for i = 1:size(newtestset,1)
    cnt_dig(newtestset(i,1)+1) = cnt_dig(newtestset(i,1)+1) + 1;
    if res(i)== newtestset(i,1)
       cnt_res(res(i)+1) = cnt_res(res(i)+1)+1;
   end
end
    
for i = 1:10
    X = ['success ratio of ',num2str(i-1)];
    disp(X);
    disp(cnt_res(i)/cnt_dig(i)*100);
end

%% Bhma 10 - A priori propabilities

%newtestset  = importdata('test.txt');  % pairnw ta test data se morfh pinaka size(newtestset,1)X257
%train = importdata('train.txt'); % pairnw ta train data 

apriori    = zeros(10,1);   % array of a-priori probabilities
train_size = size(newtrainset,1); % upologismos mege8ous train data

for i = 1:10
    apriori(i) = size(find(newtrainset(:,1)==(i-1)),1)/train_size;
end

%% Bhma 11 - Bayesian Classification

% apo bhma 7 --> meses times kai diaspores sto E_tot_all kai V_tot_all

for digit = 0:9
        meanval(:,digit+1) = reshape(E_tot_all(:,:,digit+1), 256,1);
        varval(:,digit+1)  = reshape(V_tot_all(:,:,digit+1), 256,1)+0.001;
end
%meanval = reshape(E_tot_all,[256,10]);
%varval  = reshape(V_tot_all,[256,10]);

CorrectDigitClassifications = zeros(10,1)';
aposteriori = zeros(10,1);
gauss   = zeros(256,10);

successcnt = 0;
bayesian_res = zeros(size(newtestset,1),1);
for i = 1:size(newtestset,1)
    digitchar = newtestset(i,2:257)';
    for j = 0:9
       gauss(:,j+1)  = ((1/sqrt(2*pi))*(varval(:,j+1).^(-1))).*(exp(-0.5*((digitchar-meanval(:,j+1)).^2)./(varval(:,j+1))));
       aposteriori(j+1)= sum(log(gauss(:,j+1))) + log(apriori(j+1));
       aposteriori = aposteriori';
    end
    maxaposteriori = max(aposteriori);
    flag = 1;
    for k = 1:10
        if (flag==1 && aposteriori(k) == maxaposteriori)
            classification = k - 1 ;
            flag = 0;
        end
    end
    bayesian_res(i) = classification;
    if (classification == newtestset(i,1))
        successcnt = successcnt + 1;
        CorrectDigitClassifications(newtestset(i,1)+1) =  CorrectDigitClassifications(newtestset(i,1)+1) +1;
    end
    
end

bayesratio = 100*(successcnt/size(newtestset,1));

for i = 1:10
    X = ['Naive Bayesian Success Ratio of ',num2str(i-1)];
    disp(X);
    disp(CorrectDigitClassifications(i)/cnt_dig(i)*100);
end


%% Bhma 12 - Bayesian Classification with variance = 1;
CorrectDigitClassifications_0 = zeros(10,1)';
aposteriori_0 = zeros(10,1);
gauss_0  = zeros(256,10);
successcnt_0 = 0;
varval_0 = 1;%ones(size(varval,1),size(varval,2));
bayesian_res0 = zeros(size(newtestset,1),1);
for i = 1:size(newtestset,1)
    digitchar_0 = newtestset(i,2:257)';
    for j = 0:9
       gauss_0(:,j+1)  = ((1/sqrt(2*pi))*(varval_0.^(-1))).*(exp(-0.5*((digitchar_0-meanval(:,j+1)).^2)./(varval_0)));
       aposteriori_0(j+1)= sum(log(gauss_0(:,j+1))) + log(apriori(j+1));
       aposteriori_0 = aposteriori_0';
    end
    maxaposteriori_0 = max(aposteriori_0);
    flag_0 = 1;
    for k = 1:10
        if (flag_0==1 && aposteriori_0(k) == maxaposteriori_0)
            classification_0 = k - 1 ;
            flag_0 = 0;
        end
    end
    bayesian_res0(i) = classification_0;
    if (classification_0 == newtestset(i,1))
        successcnt_0 = successcnt_0 + 1;
        CorrectDigitClassifications_0(newtestset(i,1)+1) =  CorrectDigitClassifications_0(newtestset(i,1)+1) +1;
    end
    
end

bayesratio_0 = 100*(successcnt_0/size(newtestset,1));
X = ['Naive Bayesian (with Variance equal to 1) Success Ratio : ',num2str(bayesratio_0)];
disp(X);
fprintf('\n');

for i = 1:10
    X = ['Naive Bayesian (with Variance equal to 1) Success Ratio of ',num2str(i-1)];
    disp(X);
    disp(CorrectDigitClassifications_0(i)/cnt_dig(i)*100);
end


%% Bhma 13 - NNR=1 (train = 1000, test = 100);

ratio = 0 ;
distance = zeros(1,1000);

for i = 1:100
        for j = 1:1000
            distance(j) = sqrt(sum((newtestset(i, 2:257)-newtrainset(j, 2:257)).^2)); 
        end
        
        minimum = min(distance);
        label   = find(distance == minimum); 
        
        if (newtrainset(label,1) == newtestset(i,1)) 
              ratio = ratio +1;
        end
end
disp('NNR1 Ratio for 1000 train data and 100 test data:');
disp(ratio);


%% Bhma 14 - Total Classification Using NNR-1 Algorithm
%a,b
close all;
%tic;
ratio_total_1 = 0 ;
%distance_total = zeros(size(test,1),size(train,1));
distances = zeros(size(newtestset,1),size(newtrainset,1));
NNR1_res = zeros(size(newtestset,1),1);
NNR1_res_digit = zeros(10,1);
for i = 1:size(newtestset,1) %PARFOR
        [distances(i,:),label(i)] = min_dist_label(newtestset(i,2:257),newtrainset);
        NNR1_res(i) = newtrainset(label(i),1);
        if (newtrainset(label(i),1) == newtestset(i,1)) 
              ratio_total_1 = ratio_total_1 +1; 
              NNR1_res_digit(newtestset(i,1)+1) = NNR1_res_digit(newtestset(i,1)+1)+1;
        end
end
disp('NNR1 Ratio of Total Classification:');
disp(ratio_total_1/size(newtestset,1)*100);
%toc; %to see the time of our parfors
close all;
%c
%tic;
neighbours = 5;
ratio_total = 0 ;
ratio_array_NNR_k = zeros(floor(neighbours/2),1);
for k=3:2:neighbours
    parfor i = 1:size(newtestset,1)
            number(i) = NNR_k(newtestset(i,2:257),newtrainset,k,distances(i,:));
            if (number(i) == newtestset(i,1)) 
                  ratio_total = ratio_total +1;
            end
    end
    %X = ['NNR1 Ratio of Total Classification with ',num2str(k),' neighbour(s)):'];
    %disp(X)
    %disp(ratio_total/size(test,1)*100);   
    ratio_array_NNR_k(floor(k/2),1) = ratio_total/size(newtestset,1)*100;
    ratio_total = 0;
end
figure();
plot(3:2:neighbours,ratio_array_NNR_k);
title('Success Ratio with k Neighbours')
xlabel('Neighbours')
ylabel('Success Ratio')
%toc; %to see the time of our parfors

%d
%version 1
%tic;
neighbours = 5;
ratio_total = 0 ;
ratio_array_NNR_k_impr1 = zeros(floor(neighbours/2),1);
ratio_NNR_per_digit = zeros(10,1);
for k=3:2:neighbours
    parfor i = 1:size(newtestset,1)
            number(i) = NNR_k_improved1(newtestset(i,2:257),newtrainset,k,distances(i,:));
            if (number(i) == newtestset(i,1)) 
                  ratio_total = ratio_total +1;
            end
    end
    %X = ['NNR1 Ratio of Total Classification with ',num2str(k),' neighbour(s)):'];
    %disp(X)
    %disp(ratio_total/size(test,1)*100);   
    ratio_array_NNR_k_impr1(floor(k/2),1) = ratio_total/size(newtestset,1)*100;
    ratio_total = 0;
end
figure();
plot(3:2:neighbours,ratio_array_NNR_k_impr1);
title('Success Ratio with k Neighbours using mean value of k Neighbours distances')
xlabel('Neighbours')
ylabel('Success Ratio')
%toc; %to see the time of our parfors

%result comparison
figure();
plot(3:2:neighbours,ratio_array_NNR_k_impr2-ratio_array_NNR_k);
title('Success Ratio difference between NNR-k and NNR-k with mean value of distance')
xlabel('Neighbours')
ylabel('Success Ratio')

figure();
plot(3:2:neighbours,ratio_array_NNR_k_impr2-ratio_array_NNR_k);
title('Success Ratio difference between NNR-k and NNR-k with weights')
xlabel('Neighbours')
ylabel('Success Ratio')



close all;


%%Bhma15 

traindata = newtrainset(:,2:257); %remove the infos of the correct numbers
correct_classes = newtrainset(:,1); %
%svmtrain(traindata,group,name1,value1,....,nameN,valueN) 
for i = 0:9
    myclass = correct_classes;
    myclass(myclass ~= i) = -1;
    options = statset('maxiter', inf);
    SVMStruct_lin(i+1) = svmtrain(traindata ,myclass, 'kernel_function', 'linear','kktviolationlevel',0.03, 'options', options, 'boxconstraint',0.5);
    SVMStruct_pol(i+1) = svmtrain(traindata ,myclass, 'kernel_function', 'polynomial','kktviolationlevel',0.03,'options', options, 'boxconstraint',0.5);
end

Classify_lin = zeros(size(newtestset,1),10);
Classify_lin_scores = zeros(size(newtestset,1),10);
Classify_lin_res = zeros(size(newtestset,1),1);
Classify_pol = zeros(size(newtestset,1),10);
Classify_pol_scores = zeros(size(newtestset,1),10);
Classify_pol_res = zeros(size(newtestset,1),1);

for  i = 0:9
    [Classify_lin(:,i+1),Classify_lin_scores(:,i+1)] = mysvmclassify(SVMStruct_lin(i+1),newtestset(:,2:257)); %modified classify so as to take the distances
    [Classify_pol(:,i+1),Classify_pol_scores(:,i+1)] = mysvmclassify(SVMStruct_pol(i+1),newtestset(:,2:257));
end

Classify_lin_ratio = 0;
Classify_lin_ratio_per_digit = zeros(10,1);
Classify_pol_ratio = 0;
Classify_pol_ratio_per_digit = zeros(10,1);

for i = 1:size(newtestset,1)
    [~,num] = min(Classify_lin_scores(i,:));
    Classify_lin_res(i) = num-1;
    if ( newtestset(i,1) == (num-1)) 
        Classify_lin_ratio = Classify_lin_ratio + 1;
        Classify_lin_ratio_per_digit(newtestset(i,1)+1) = Classify_lin_ratio_per_digit(newtestset(i,1)+1) + 1;
    end
    [~,num] = min(Classify_pol_scores(i,:));
    Classify_pol_res(i) = num-1;
    if ( newtestset(i,1) == (num-1)) 
        Classify_pol_ratio = Classify_pol_ratio + 1;
        Classify_pol_ratio_per_digit(newtestset(i,1)+1) = Classify_pol_ratio_per_digit(newtestset(i,1)+1) + 1;
    end    
end

%Classify_lin_ratio = 100*(Classify_lin_ratio/size(newtestset,1));
X = ['Svm with Linear Kernel Success Ratio : ',num2str(100*(Classify_lin_ratio/size(newtestset,1)))];
disp(X);
fprintf('\n');

for i = 1:10
    X = ['Svm with Linear Kernel success ratio of ',num2str(i-1)];
    disp(X);
    disp(Classify_lin_ratio_per_digit(i)/cnt_dig(i)*100);
    SVM_with_linear_kernel_Classifier_Success_Rate_per_digit(i) = (Classify_lin_ratio_per_digit(i)/cnt_dig(i)*100);
end
save('SVM_with_linear_kernel_Classifier_Success_Rate_per_digit.mat','SVM_with_linear_kernel_Classifier_Success_Rate_per_digit');

%Classify_pol_ratio = 100*(Classify_pol_ratio/size(newtestset,1));
X = ['Svm with Polynomial Kernel Success Ratio : ',num2str(100*(Classify_pol_ratio/size(newtestset,1)))];
disp(X);
fprintf('\n');

for i = 1:10
    X = ['Svm with Polynomial Kernel success ratio of ',num2str(i-1)];
    disp(X);
    disp(Classify_pol_ratio_per_digit(i)/cnt_dig(i)*100);
    SVM_with_polynomial_kernel_Classifier_Success_Rate_per_digit(i) = (Classify_pol_ratio_per_digit(i)/cnt_dig(i)*100);
end
save('SVM_with_polynomial_kernel_Classifier_Success_Rate_per_digit.mat','SVM_with_polynomial_kernel_Classifier_Success_Rate_per_digit');

%%Fit discriminant analysis classifier
Fit_Discriminant = fitcdiscr(traindata,correct_classes);
Fit_discr_out = predict(Fit_Discriminant, newtestset(:,2:257));
Classify_fit_discr_ratio = 0;
Classify_fit_discr_ratio_per_digit = zeros(10,1);
Classify_fit_discr_res = zeros(size(newtestset,1),1);
for i= 1:size(newtestset,1)
    Classify_fit_discr_res(i) = Fit_discr_out(i);
    if (Fit_discr_out(i) == newtestset(i,1))
        Classify_fit_discr_ratio = Classify_fit_discr_ratio+1;
        Classify_fit_discr_ratio_per_digit(newtestset(i,1) + 1) = Classify_fit_discr_ratio_per_digit(newtestset(i,1) + 1) + 1;
    end;
end

%Classify_fit_discr_ratio = 100*(Classify_fit_discr_ratio/size(test,1));
X = ['Fit discriminant analysis classifier Success Ratio : ',num2str(100*(Classify_fit_discr_ratio/size(newtestset,1)))];
disp(X);
fprintf('\n');

for i = 1:10
    X = ['Fit discriminant analysis classifier success ratio of ',num2str(i-1)];
    disp(X);
    disp(Classify_fit_discr_ratio_per_digit(i)/cnt_dig(i)*100);
    Fit_discriminant_analysis_Classifier_Success_Rate_per_digit(i) = (Classify_fit_discr_ratio_per_digit(i)/cnt_dig(i)*100);
end
save('Fit_discriminant_analysis_Classifier_Success_Rate_per_digit.mat','Fit_discriminant_analysis_Classifier_Success_Rate_per_digit');

%%Bhma 16
%a
classificator_ratio(1,:) = cnt_res(:,1) ./ cnt_dig(:,1); %Euclidean Classifier
classificator_ratio(2,:) = CorrectDigitClassifications(:) ./ cnt_dig(:,1); %Bayesian Classifier
classificator_ratio(3,:) = CorrectDigitClassifications_0(:) ./ cnt_dig(:,1); %Bayesian Classifier with variance = 1
classificator_ratio(4,:) = NNR1_res_digit(:) ./ cnt_dig(:,1); %NNR - 1 Classifier
classificator_ratio(5,:) = Classify_lin_ratio_per_digit(:) ./ cnt_dig(:,1); %SVM with linear kernel Classifier
classificator_ratio(6,:) = Classify_pol_ratio_per_digit(:) ./ cnt_dig(:,1); %SVM with polynomial kernel Classifier
classificator_ratio(7,:) = Classify_fit_discr_ratio_per_digit(:) ./ cnt_dig(:,1); %Fit discriminant analysis Classifier

classificator_results(1,:) = euclidean_res; %Euclidean Classifier
classificator_results(2,:) = bayesian_res; %Bayesian Classifier
classificator_results(3,:) = bayesian_res0;%Bayesian Classifier with variance = 1
classificator_results(4,:) = NNR1_res; %NNR - 1 Classifier
classificator_results(5,:) = Classify_lin_res; %SVM with linear kernel Classifier
classificator_results(6,:) = Classify_pol_res ; %SVM with polynomial kernel Classifier
classificator_results(7,:) = Classify_fit_discr_res; %Fit discriminant analysis Classifier

%find the most frequent value of each column of classificator results'
%array
%https://www.mathworks.com/help/matlab/ref/mode.html

%close all;
toc;

%b
size (Classify_lin_scores)
size (Classify_pol_scores)
%Normalize SVM scores
for i=1:size(newtestset,1)
    if (min(Classify_lin_scores(i,:)) > 0)
        Classify_lin_scores2(i,:) = Classify_lin_scores(i,:) - abs(min(Classify_lin_scores(i,:)));
    else
        Classify_lin_scores2(i,:) = Classify_lin_scores(i,:) + abs(min(Classify_lin_scores(i,:)));
    end
    Classify_lin_scores2(i,:) = Classify_lin_scores2(i,:)/max(Classify_lin_scores2(i,:));  
    
    if (min(Classify_pol_scores(i,:)) > 0)
        Classify_pol_scores2(i,:) = Classify_pol_scores(i,:) - abs(min(Classify_pol_scores(i,:)));
    else
        Classify_pol_scores2(i,:) = Classify_pol_scores(i,:) + abs(min(Classify_pol_scores(i,:)));
    end
    Classify_pol_scores2(i,:) = Classify_pol_scores2(i,:)/max(Classify_pol_scores2(i,:));  
end
%Calculate scores judging by distance
%find the mean distance for each test_data from each digit
metric = zeros(size(newtestset,1),10);
for i = 1:size(newtestset,1)
    for j = 0:9
        tempdist = distances(i, newtrainset(:,1)==j);
        metric(i, j+1) = min(tempdist);
        %metric(i,j+1) = 1/metric(i,j+1);
    end
end

for i = 1: size(newtestset,1)
    metric(i,:) = metric(i,:) / max(metric(i,:));
end

disp('---------------------------------- Step 16b ----------------------------------');
MaxFunc = 0;
MinFunc = 0;
ProdFunc = 0;
MeanFunc = 0;

for i = 1:size(newtestset,1)
    lin_scores = Classify_lin_scores2(i,:);
    pol_scores = Classify_pol_scores2(i,:);
    dist_scores= metric(i,:);
    temp = [lin_scores ; pol_scores ; dist_scores];
    [maxtemp, resmax] = max(max(temp));
    [mintemp, resmin] = max(min(temp));
    temp2 = temp;
    temp2(isnan(temp2)) = 0.001;
    [prodtemp, resprod] = max(prod(temp2));
    temp3 = temp;
    temp3(isnan(temp3)) = 0.000;
    [meantemp, resmean] = max(sum(temp3)/4);

    if (resmax -1 == newtestset(i,1))
        MaxFunc = MaxFunc + 1;
    end
    if (resmin -1 == newtestset(i,1))
        MinFunc = MinFunc + 1;
    end
    if (resprod -1 == newtestset(i,1))
        ProdFunc = ProdFunc + 1;
    end
    if (resmean -1 == newtestset(i,1))
        MeanFunc = MeanFunc + 1;
    end
end
successRateMax = MaxFunc/size(newtestset,1);
successRateMin = MinFunc/size(newtestset,1);
successRateProd = ProdFunc/size(newtestset,1);
successRateMean = MeanFunc/size(newtestset,1);    



disp('---------------------------------- Step 16c ----------------------------------');

load('trainscores.mat');
load('testscores.mat');

for i = 1:size(examineScores,1)
    temp = squeeze(examineScores(i,:,:))';
    temp(isnan(temp)) = 0;
    trainscores(i,1) = newtestset(i,1);
    trainscores(i,2:41) = reshape(temp, 1, 40);
end

for i = 1:size(examineScores2,1)
    temp = squeeze(examineScores2(i,:,:))';
    temp(isnan(temp)) = 0;
    testscores(i,1) = testdata(i,1);
    testscores(i,2:41) = reshape(temp, 1, 40);
end

disp('************************ Classifier using confidence scores as features ************************');
newtraindata = trainscores(:, 2:41);
newclasses = trainscores(:,1);
for digit = 0:9
    myclass = newclasses;
    myclass(myclass ~= digit) = -1;
    options = statset('maxiter', inf);
    SVMStruct(digit+1) = svmtrain(newtraindata ,myclass, 'kernel_function', 'linear', 'options', options);
end

for digit = 0:9
    GroupC(:,digit+1) = svm_classify(SVMStruct(digit+1),testscores(:,2:41))';
end

correctClassificationsLinearSVM = 0;
for i= 1:size(testscores,1)
    whichDigitLinSVM(i) = (find(GroupC(i,:) == min(GroupC(i,:)))-1);
    if (whichDigitLinSVM(i) == testscores(i,1))
        correctClassificationsLinearSVM = correctClassificationsLinearSVM+1;
    end;
end
successRateLinearSVM = correctClassificationsLinearSVM/size(testscores,1);
fprintf('Success Rate: %f\n', successRateLinearSVM);

    