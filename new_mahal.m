%Load data
smooth_all = load('D:\ABG_Projects_Backup\ssn_modelling\ssn-simulator\results\11-12\noise_end_stair_stimuli_noise200.0gE0.3lamda1\noisy_responses_train_untrain_control\noisy_response_.mat');

%Specify number of epochs (pre and post training)
num_epochs = 2;

%Specify number of layers (middle and superficial)
num_layers = 2;

%Specify number of components to use
PC_used = 15;

labels = smooth_all.labels(1:900);

%Initialise empty cells to store results
finalSD_results = cell(num_epochs,1);
finalME_results = cell(num_epochs,1);
finalSNR_results = cell(num_epochs,1);
centre_E_indices =[20 21 22 23 24 29 30 31 32 33 38 39 40 41 42 47 48 49 50 51 56 57 58 59 60];

%Iterate over number of epochs
for epoch =1:num_epochs

    %Specify layer (middle/superficial)!!!
    curr_data = squeeze(smooth_all.superficial(epoch, :, :));

    %Normalise data
    Curr_data_used = zscore(curr_data);

    %PCA
    [coeff,score,latent,tsquared,explained,mu] = pca(curr_data);

    %Select components
    Curr_data_used_pca = score(:,1:PC_used);

    %Separate data into orientation conditions 
    train_data   = Curr_data_used_pca(find(labels ==55), :);
    untrain_data = Curr_data_used_pca(find(labels ==125), :);
    control_data = Curr_data_used_pca(find(labels ==0), :); 

    %Calculate Mahalanobis distance
    train_dis_mahal   = sqrt(mahal(train_data,control_data));
    untrain_dis_mahal = sqrt(mahal(untrain_data,control_data));

    train_dis_mahal_forSNR   = train_dis_mahal;
    untrain_dis_mahal_forSNR = untrain_dis_mahal;

    %Mean
    train_dis_mahal_mean   = mean(train_dis_mahal);
    untrain_dis_mahal_mean = mean(untrain_dis_mahal);

    % step 2 calculate the sd
    train_data_size = size(train_data,1);
    distanceSD_train = zeros(train_data_size,1);
    distanceSD_untrain = zeros(train_data_size,1);
    distanceSD_control = zeros(train_data_size,1);
    for cross_size_i = 1:train_data_size
        train_data_temp   = train_data;
        untrain_data_temp = untrain_data;
        control_data_temp = control_data;

        train_data_temp(cross_size_i,:) = [];
        untrain_data_temp(cross_size_i,:) = [];
        control_data_temp(cross_size_i,:) = [];

        distanceSD_train(cross_size_i,1) = sqrt(mahal(train_data(cross_size_i,:),train_data_temp));
        distanceSD_untrain(cross_size_i,1) = sqrt(mahal(untrain_data(cross_size_i,:),untrain_data_temp));
        distanceSD_control(cross_size_i,1) = sqrt(mahal(control_data(cross_size_i,:),control_data_temp));  
    end                                     

    distanceSD_train_forSNR =  distanceSD_train;
    distanceSD_untrain_forSNR = distanceSD_untrain;
    distanceSD_control_forSNR = distanceSD_control;

    %remove outliers
    %distanceSD_train_z = zscore(distanceSD_train);
    %distanceSD_train(abs(distanceSD_train_z)>outlier_threshold) = [];
    %                     
    %distanceSD_untrain_z = zscore(distanceSD_untrain);
    %distanceSD_untrain(abs(distanceSD_untrain_z)>outlier_threshold) = [];

    train_dis_mahal_sd   = mean(distanceSD_train);
    untrain_dis_mahal_sd = mean(distanceSD_untrain);
    control_dis_mahal_sd = mean(distanceSD_control); 

    % step 3 calculate the SNR
    train_SNR   = train_dis_mahal_forSNR   ./ distanceSD_train_forSNR;
    untrain_SNR = untrain_dis_mahal_forSNR ./ distanceSD_untrain_forSNR;                                      
    train_SNR_mean   = mean(train_SNR);
    untrain_SNR_mean = mean(untrain_SNR);


    %Plot distances
    figure(1)
    subplot(2,1,1);
    histogram(train_dis_mahal);
    title('Trained ori')
    subplot(2,1,2);
    histogram(untrain_dis_mahal);
    title('Untrained ori')

    finalME_results{epoch} = [train_dis_mahal_mean,untrain_dis_mahal_mean];
    finalSD_results{epoch} = [train_dis_mahal_sd,untrain_dis_mahal_sd,control_dis_mahal_sd];
    finalSNR_results{epoch} = [train_SNR_mean,untrain_SNR_mean];
    
end
