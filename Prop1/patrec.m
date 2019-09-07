% Ergasthriaki Askhsh 1 - Proparaskeuh
% Sunergates : 
%              Vavouliotis Georgios  (03112083)
%              Stavrakakis Dimitrios (03112017)


close all; clear all; clc ;

%% Bhma 1 -  Sxediasmos tou 131ou psifiou

train    = importdata('train.txt'); % pairnw ta train data se morfh pinaka 7291X257
 
digit131 = train(131,2:257); % pairnw ta xaraktiristika toy 131ou psifiou

matrix = reshape(digit131, [16, 16]); % ta organwnw se ena pinaka 16x16

% emfanizw to psifio auto kai to apo8hkeuw katallhla
mkdir Bhma1Results ;
cd Bhma1Results ;
figure(1); 
imagesc(matrix);
title('Plot1-Digit131');
print -djpeg Plot131.jpeg; 
cd ../

%% Bhma 2,3 - Mesh Timh & Diaspora xarakthristikwn gia to pixel (10,10) tou 0 sta train data

sum1 = 0;    % krataei to a8roisma gia ta (10,10)
cnt  = 0;    % metarei ta mhdenika
e2new   = 0; % krataei to a8roisma twn tetragwnwn gia ta (10,10) gia na vrw th diaspora

for i = 1:7291 
    if train(i,1) == 0 % an prokeitai gia 0
        zero = train(i,2:257); % pairnw ta xaraktiristika toy
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
for i = 1:7291 
    if train(i,1) == 0 % an vrw mhdeniko kanw oti kai sto bhma 2,3 
        zero     = train(i,2:257);
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
mkdir Bhma5-6Results;
cd Bhma5-6Results
figure(2);
imagesc(E_tot);
title('Zero using Mean Value');
print -djpeg Zero_Mean.jpeg;
cd ../

%% Bhma 6 - Sxediasmos tou 0 me bash th diaspora
cd Bhma5-6Results ;
figure(3);
imagesc(V_tot);
title('Zero using Variance');
print -djpeg Zero_Var.jpeg;
cd ../

%% Bhma 7 - Ypologismos E,V gia ola ta digits me xrhsh twn train data

sum_tot_all = zeros(16,16,10); % gia ka8e digit krataei to a8roisma twn pixel
e_2_all     = zeros(16,16,10); % gia ka8e digit krataei to a8roisma twn tetragwnwn twn pixel
cnt         = zeros(10);       % krataei poses fores vrhka ka8e digit gia na vrw th mesh timh

for i = 1:7291 
    k = train(i,1)+1; % einai o deikths gia tou pou 8a kanw store(to +1 ofeiletai sto oti oi pinakes sto matlab arxizoun thn ari8misi apo to 1) 
    % akolou8w thn idia diadikasia me prin
    zero = train(i,2:257);
    arr  = reshape(zero,[16,16]);
    sum_tot_all(:,:,k)  = sum_tot_all(:,:,k) + arr;
    e_2_all(:,:,k)      = e_2_all(:,:,k) + arr.^2;
    cnt(k)  = cnt(k) +1;
end

% Sxediasmos olwn twn digits me xrhsh ths meshs timhs

% pinakes meshs timhs kai diasporas gia ka8e digit 
E_tot_all = zeros(16,16,10);
V_tot_all = zeros(16,16,10);
mkdir Bhma7Results ;
save_path = './Bhma7Results/%d.jpg' ;

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

test = importdata('test.txt'); % pairnw ta train data se morfh pinaka 2007X257

digit101 = test(101,2:257); % pairnw ta xaraktiristika toy 101ou psifiou

matrix = reshape(digit101, [16, 16]); % ta organwnw se ena pinaka 16x16

% emfanizw to psifio auto kai to apo8hkeuw katallhla
mkdir Bhma8Results ;
cd Bhma8Results;
figure(65); 
imagesc(matrix);
title('Plot-Digit101');
print -djpeg Plot101.jpeg; 
cd ../

% euresh eukleidias apostashs gia kaue digit wste na kanoume swsta to
% classification

dist = zeros(16,16,10); % pinakas eukleidiwn apostasewn
sums = zeros(10,1);     % gia ka8e digit exei to a8roisma twn eukleidiwn apostasewn
%D = zeros(10,1);
for k = 1:10
    %D(k) = sum(sum(pdist2(matrix,E_tot_all(:,:,k),'euclidean')));
    dist(:,:,k) = (matrix(:,:) - E_tot_all(:,:,k)).^2;
    sums(k) = sum(sum(dist(:,:,k)));
end

% briskw to elaxisto a8roisma eukleidiwn apostasewn
[~,index] = min(sums(:));
disp('Step 8 --> Digit 101 of test data result = ');
mynumber = index-1 ; % brhka to digit pou o taksinomitis mou kanei match me to 101o pshfio
disp(mynumber);

%% Bhma 9 - Taksinomisi olwn twn pshfiwn twn test data se mia apo tis 10 kathgories me xrhsh eukleidias apostashs 

res = zeros(2007,1); % o pinakas autos 8a periexei to apotelesma tou taksinomiti mou gia ta test data

for i = 1:2007 
    k = test(i,1)+1;  % einai o deikths gia tou pou 8a kanw store(to +1 ofeiletai sto oti oi pinakes sto matlab arxizoun thn ari8misi apo to 1) 
    
    % akolou8w th diadikasia tou Bhmatos 8
    zero = test(i,2:257);
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

% Success rate

cnt1 = 0;
for i = 1:2007
   if res(i)== test(i,1)
       cnt1=cnt1+1;
   end
end

disp('Step 9 : Success Rate =');
Ratio = (cnt1/2007)*100;
disp(Ratio);

% Success rate for each digit
cnt_res = zeros(10);
cnt_dig = zeros(10);

for i = 1:2007
    cnt_dig(test(i,1)+1) = cnt_dig(test(i,1)+1) + 1;
    if res(i)== test(i,1)
       cnt_res(res(i)+1) = cnt_res(res(i)+1)+1;
   end
end
    
for i = 1:10
    X = ['success ratio of ',num2str(i-1)];
    disp(X);
    %disp(i)
    disp(cnt_res(i)/cnt_dig(i)*100);
end


