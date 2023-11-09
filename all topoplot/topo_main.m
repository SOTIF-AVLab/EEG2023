subject_list = {'fujunwen', 'gongxuechun', 'liuchangjie', 'liulonglong', 'liuwei', 'liuxiaochuan', 'maocui', 'murui', 'wuyijia', 'zhangxiaofei', 'zhaochengxiang'};
% 

current_dir = pwd;
folder_path = [current_dir, '\..\..\Result\topoplot_100\']; 
for subject_idx = 1:numel(subject_list)
    
    new_dir_path = [folder_path, char(subject_list(subject_idx))];
    mkdir(new_dir_path);
    subject = subject_list{1, subject_idx};
    for i = 1:15

        topographic_plot(subject, i);
    end
end
