
clear;
close all;
clc
%#############################Read Data##############################################################%
Current_Position=pwd;
Raw_Data=[Current_Position,'\..\..\Processed_Data\fjw'];
Fine_Data=[Current_Position,'\..\..\Processed_Data\fjw\Fine_Data'];
ERP_Data=[Current_Position,'\..\..\Processed_Data\fjw\ERP'];

Raw_Data_aa=dir(Raw_Data);
Name_People={Raw_Data_aa.name};
Position_Raw_Data=Raw_Data;
data_filenames={'1_1_segmented_data.set', '1_2_segmented_data.set' ...
    '2_1_segmented_data.set', '2_2_segmented_data.set', '3_1_segmented_data.set'...
    '3_2_segmented_data.set','4_1_segmented_data.set','4_2_segmented_data.set', '5_1_segmented_data.set'...
    '5_2_segmented_data.set', '6_1_segmented_data.set','6_2_segmented_data.set','7_1_segmented_data.set'...
    '7_2_segmented_data.set','8_1_segmented_data.set','8_2_segmented_data.set'};

%######################################################################################################%
% Current_Position=pwd;
for event_num = [1 3:14 100 20]
%#############################Define data##############################################################%
%####################################Extract Events###################################%
ALLEEG = {};
for i = 1:length(data_filenames)
    Data_Postion=[Position_Raw_Data,'\',data_filenames{i}];
    ALLEEG = [ALLEEG, pop_loadset(Data_Postion)];
end

close all
% Select epochs for a specific event
event_type = event_num; % Set the event type you want to select, cut in from left
epoch_start = -0.2; % Set the epoch start time in seconds
epoch_end = 1; % Set the epoch end time in seconds
ERP_Figure_Position=[ERP_Data,'\',Name_People{1,3},'\event'];
Fine_Data_Position=[Fine_Data,'\event'];

mkdir([ERP_Figure_Position,num2str(event_type)]);
mkdir([Fine_Data_Position,num2str(event_type)]);


ERP_Data_Position_Event=[ERP_Figure_Position,num2str(event_type)];
Fine_Data_Position_Event=[Fine_Data_Position,num2str(event_type)];

% Fine_Data_Position_Event=[Fine_Data_Position,num2str(event_type)];
% mkdir([Fine_Data_Position,num2str(event_type)])
% Initialize list of selected datasets
ALLEEG_selected = {};

% Loop over all datasets
for i = 1:length(ALLEEG)
    % Check if the event type is present in this dataset
    event_idx = [];
    %   event_idx = find([ALLEEG{1,1}.epoch.eventtype{1:33}] == 7);

    ALLEVENT = {ALLEEG{1,i}.epoch.eventtype};
    has_event_type = 0;
    for j = 1:length(ALLEVENT)
    % Check if the dataset contains the event_type
    if iscell(ALLEVENT{1,j})
        if ~isempty(ALLEVENT{1,j})
            for count = 1: length(ALLEVENT{1,j})
                if ALLEVENT{1,j}{1,count} == event_type
                    has_event_type = 1;
                    break;
                else
                    continue;
                end
            end
        else
            continue;
        end
    elseif isnumeric(ALLEVENT{1,j})
        if ALLEVENT{1,j} == event_type
            has_event_type = 1;
            break;
        else
            continue;
        end
    end
    end
    if has_event_type == 1
    [ALLEEG{1,i}, ~, ~] = pop_epoch(ALLEEG{1,i}, {event_type}, [epoch_start epoch_end]);
    % Add to selected dataset list
    ALLEEG_selected{end+1} = ALLEEG{i};
    else
        i = i+1;
    end
end

% Check if any datasets were selected
if isempty(ALLEEG_selected)
    error('None of the input datasets contain the specified event type');
end

% Get the channel labels that are common to all datasets
common_ch_labels = {ALLEEG{1,1}.chanlocs.labels};
for i = 2:length(ALLEEG)
    common_ch_labels = intersect(common_ch_labels, {ALLEEG{1,i}.chanlocs.labels});
end

% Get the channel indices corresponding to the common labels
common_ch_idx = [];
for i = 1:length(common_ch_labels)
    for j = 1:length(ALLEEG_selected)
    common_ch_idx(j,i) = find(strcmp({ALLEEG_selected{1,j}.chanlocs.labels}, common_ch_labels{i}));
    end
end

%#######################################save data#############################%
% Preallocate matrix to store all EEG data
n_channels = length(common_ch_idx(1,:));
n_timepoints_all = 0; % initialize to zero
for i = 1:length(ALLEEG)
    n_timepoints_all = n_timepoints_all + length(ALLEEG{1,i}.times);
end
eegdata_all = zeros(n_channels, n_timepoints_all);

common_ch = ALLEEG_selected{1,1}.chanlocs(common_ch_idx(1,:));

EEG_data = [];

% Loop over all datasets in ALLEEG
for i = 1:length(ALLEEG_selected)
    for j = 1:n_channels
    % Extract the data from the common channels for the current dataset
    eegdata(j,:,:) = ALLEEG_selected{1,i}.data(common_ch_idx(i,j), :, :);
    if size(eegdata,2) ~= (epoch_end - epoch_start)*1000
        eegdata = permute(eegdata, [1 3 2]);
    end
    end
    % Concatenate the data along the third dimension
    EEG_data = cat(3, EEG_data, eegdata);
    eegdata = [];
 end

% Average across epochs
ERP = squeeze(mean(EEG_data, 3));
ERP_Fine_Data_Position=[Fine_Data_Position_Event,'\ERP of even']
save([ERP_Fine_Data_Position,num2str(event_type)],'ERP');

% Plot topography at specific time points
timepoints = -200:50:1000;
for i = 1:length(timepoints)
    [~, time_idx] = min(abs(ALLEEG_selected{1,1}.times - timepoints(i)));
    figure;
    topoplot(ERP(:, time_idx), common_ch,'maplimits',[-10 10]);
    colorbar
    caxis([-10 10]);
    title(['Topography at ', num2str(ALLEEG_selected{1,1}.times(time_idx)), ' ms']);
    filename = [ERP_Data_Position_Event,'\Topography at ', num2str(ALLEEG_selected{1}.times(time_idx)), ' ms'];
    saveas(gcf,filename,'png');
end
%%
close all
legends = {'FZ','CZ','P7','P3','PZ','P4','P8','OZ'};
for i = 1:length(legends)
   curr_idx = find(strcmp(common_ch_labels, legends{i}));
         % Store the index in selected_ch_idx if the label is found
        if ~isempty(curr_idx)
            selected_ch_idx(i) = curr_idx;
        end
end
figure;
for j = 1:length(selected_ch_idx)
        i = selected_ch_idx(j);
        if i ~= 0
        plot(-200:999,ERP(i,:))
        title(['ERP at selected channels',legends(j)]); 
        legend(legends(j));
%     xline(300); 
%         ylim([-10 10]);
        filename = cell2mat([ERP_Data_Position_Event,'\ERP at selected channels',legends(j)]);
        saveas(gcf,filename,'png');
        else
            continue
        end
end

close all
down_sampling_factor = 4; % downsampling to 250 hz
EEG_data_ds = EEG_data(:,1:down_sampling_factor:end,:);
filename = [Fine_Data_Position_Event,'\event_data',num2str(event_type),'.mat'];
save(filename,'EEG_data_ds')
% clear
end