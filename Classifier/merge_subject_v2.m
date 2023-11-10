clear
close all
clc



% Loop through subjects from 1 to 12
for subjectNumber = [1:8 10:12]
    window_length = 2000:4999;
    Current_path = pwd;
    data_path = [Current_path, '\..\Data\'];
    folderName = [data_path, 'subject', num2str(subjectNumber)];
    mkdir(folderName);

    % Load the available EEG data for the subject
    load(['E:\EEG Data\EEG_Analysis\Dataset\subject', num2str(subjectNumber)]);
    load(['E:\EEG Data\EEG_Analysis\Dataset\subject', num2str(subjectNumber),'_non']);
    eventNumber_list = [1 2 7 8 13 14];
    non_eventNumber_list = [3 4 5 6 11 12];
    for i = 1:6
    eventNumber = eventNumber_list(i);
    non_eventNumber = non_eventNumber_list(i);
    % Determine the total number of available data samples
    total_samples = size(target_set_data{1, eventNumber}, 3); % Assuming event 1 has the same number of samples for all events
    non_total_samples = size(target_set_data{1, non_eventNumber}, 3);
    down_sampling_factor = 4; % Downsampling to 250 Hz

    % Randomly shuffle the indices of available data samples
    shuffled_indices = randperm(total_samples);
    non_shuffled_indices = randperm(non_total_samples);

    % Define the number of sets based on the available data
%     num_sets = ceil(total_samples / 100); % Each set contains 100 samples, adjust as needed

    % Loop through different sets
    for setNumber = 1:4
        if setNumber == 1
        % Extract a subset of shuffled indices for this set
        start_idx = floor((setNumber-1)*total_samples/4+1);
        end_idx = floor(total_samples/2);
        subset_indices = shuffled_indices(start_idx:end_idx);
        non_start_idx = floor((setNumber-1)*non_total_samples/4+1);
        non_end_idx = floor(non_total_samples/2);
        non_subset_indices = non_shuffled_indices(non_start_idx:non_end_idx);
        else
         % Extract a subset of shuffled indices for this set
        start_idx = floor((setNumber-2)*total_samples/6 + 1 + total_samples/2);
        end_idx = floor((setNumber-1)*total_samples/6 + total_samples/2);
        subset_indices = shuffled_indices(start_idx:end_idx);
        non_start_idx = floor((setNumber-2)*non_total_samples/6 + 1 + non_total_samples/2);
        non_end_idx = floor((setNumber-1)*non_total_samples/6 + non_total_samples/2);
        non_subset_indices = non_shuffled_indices(non_start_idx:non_end_idx);
        end

        % Extract X1 and X2 from the available data based on subset_indices
        X1 = double(target_set_data{1, eventNumber}(:,window_length,subset_indices(1:end)));
        X2 = non_target_data(:, window_length, ((setNumber-1)*100+1):(setNumber*100));
        X3 = double(target_set_data{1, non_eventNumber}(:,window_length,non_subset_indices(1:end)));
        
        % Extract unwanted channels
        X1([9 10 63],:,:)=[];
        X2([9 10 63],:,:)=[];
        X3([9 10 63],:,:)=[];
        % Down-sampling
        X1 = X1(:,1:down_sampling_factor:end,:);
        X2 = X2(:,1:down_sampling_factor:end,:);
        X3 = X3(:,1:down_sampling_factor:end,:);
        % Create a new folder for this set
        create_new_folder = [folderName, '\event', num2str(eventNumber)];
        mkdir(create_new_folder);

        % Save X1 and X2 in the new folder
        save([create_new_folder, '\sub1_', num2str(setNumber), '_data.mat'], 'X1', 'X2','X3', '-v7.3');
    end
    end
    clear
end
