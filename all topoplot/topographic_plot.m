% This file aims to plot the topographic file of the testee under specific
% type of event. 
% Can change the interested event type
function topographic_plot(subject, event)
    close all
    if event == 2 
        event = 20;
    elseif event == 15
        event = 100;
    end
    Current_Position = pwd;
    processed_data_path = [Current_Position, '\..\..\Processed_Data\', subject, '\dataset'];

    % Start EEGLAB
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

    if strcmp(subject, 'zhangxiaofei')
        data_filenames={'1_1_segmented_data_ica.set', '1_2_segmented_data_ica.set' ...
            '2_1_segmented_data_ica.set', '2_2_segmented_data_ica.set', '3_1_segmented_data_ica.set'...
            '3_2_segmented_data_ica.set','4_1_segmented_data_ica.set', '5_1_segmented_data_ica.set'...
            '5_2_segmented_data_ica.set', '6_1_segmented_data_ica.set','7_1_segmented_data_ica.set'...
            '7_2_segmented_data_ica.set','8_1_segmented_data_ica.set','8_2_segmented_data_ica.set'};
    elseif strcmp(subject, 'gongxuechun')
         data_filenames={'1_1_segmented_data_ica.set', '1_2_segmented_data_ica.set' ...
            '2_1_segmented_data_ica.set', '2_2_segmented_data_ica.set', '4_1_segmented_data_ica.set',...
            '6_2_segmented_data_ica.set', '5_1_segmented_data_ica.set'...
            '5_2_segmented_data_ica.set', '6_1_segmented_data_ica.set','7_1_segmented_data_ica.set'...
            '7_2_segmented_data_ica.set','8_1_segmented_data_ica.set','8_2_segmented_data_ica.set'};
    else 
        data_filenames={'1_1_segmented_data_ica.set', '1_2_segmented_data_ica.set' ...
            '2_1_segmented_data_ica.set', '2_2_segmented_data_ica.set', '3_1_segmented_data_ica.set'...
            '3_2_segmented_data_ica.set','4_1_segmented_data_ica.set', '4_2_segmented_data_ica.set', '5_1_segmented_data_ica.set'...
            '5_2_segmented_data_ica.set', '6_1_segmented_data_ica.set','6_2_segmented_data_ica.set', '7_1_segmented_data_ica.set'...
            '7_2_segmented_data_ica.set','8_1_segmented_data_ica.set','8_2_segmented_data_ica.set'};
    end
            
    %     Select the event types you're interested in for the topo plot
        event_type = [event]; % use your own event types here

    for i = 1: numel(data_filenames)
    %     Load each dataset
        file_path = [processed_data_path, '\' ,data_filenames{i}];
        EEG = pop_loadset(file_path);
        [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, i);

    %     Extract the indices of the epochs corresponding to the event types
        event_indices = [];
        for j = 1:size(EEG.epoch, 2)
            for k = 1:numel(event_type)
                if size(EEG.epoch, 2) == size(EEG.event, 2)
                    if EEG.epoch(j).eventtype == event_type(k)
                       event_indices = [event_indices, j];
                    end
                else
                    if EEG.epoch(j).eventtype{1} == event_type(k)
                       event_indices = [event_indices, j];
                    end
                end
            end
        end

    %     Create a new structure containing only the epochs of the desired
    %     event types
        if ~isempty(event_indices)
            EEG_event = pop_select(EEG, 'trial', event_indices);
        else
            EEG_event = [];
        end

    %     Store the EEG structure in the cell array
        EEG_event_cells{i} = EEG_event;
    end

    % manually merge the EEG datasets
    EEG_event_merged = [];
    idx = 1;
    while isempty(EEG_event_merged)
        EEG_event_merged = EEG_event_cells{idx};
        idx = idx + 1;
    end
    for i = idx:numel(EEG_event_cells)
        if ~isempty(EEG_event_cells{i})
            EEG_event_merged.epoch = [EEG_event_merged.epoch, EEG_event_cells{i}.epoch];
            EEG_event_merged.data = cat(3, EEG_event_merged.data, EEG_event_cells{i}.data);
        end
    end

    % Average the epochs for each event type
    avg_data = mean(EEG_event_merged.data, 3);

    % Create a new EEG structure for the averaged data
    EEG_event_avg = EEG_event_merged;
    EEG_event_avg.data = avg_data;
    EEG_event_avg.trials = 1;

    % Plot the topographic maps for the averaged EEG data

    time_interval = 100;
    num_plot = size(EEG_event_avg.data, 2) / time_interval;
    for i = 1:40
    %     Define the time range for the current interval. 
        start_time = (i - 1) * time_interval + 1 + 2000;
        end_time = i*time_interval + 2000;

        subplot(4, 10, i);
        EEG_interval = EEG_event_avg;
        EEG_interval.data = EEG_event_avg.data(:, start_time:end_time);
        EEG_interval.times = EEG_event_avg.times(:, start_time:end_time);
        EEG_interval.pnts = time_interval;
        topoplot(EEG_interval.data(:, 1), EEG_interval.chanlocs);
        title(sprintf('%d - %d ms', EEG_interval.times(1, 1), EEG_interval.times(1, end)));
        cbar = colorbar;
        caxis([-6 6]);
    end
    event_str = char(string(event));
    topo_path_fig = [Current_Position, '\..\..\Result\topoplot_100\', subject, '\', 'event_',  event_str,'.fig'];    
    saveas(gcf, topo_path_fig);

    
end