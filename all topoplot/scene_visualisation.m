clear all
close all
full_name = "maocui";
abv_name = "mc";
test_scene = "1_1";
test_date = "0729";
test_time = "1435";

time = linspace(-200, 199, 400);

data_path = "../../../../EEGData/data/" + test_date + "_" + full_name + "/"  + ...
            test_scene + "/" + test_date + "_" + abv_name + "_" + ...
            test_scene + ".mat";

data = load(data_path).ans.Data;

ego_x_pos_column = 218;
ego_y_pos_column = 219;

id = 1;
x_pos_column = (id - 1) * 8 + 10;
y_pos_column = (id - 1) * 8 + 11;
y_vel_column = (id - 1) * 8 + 13;

row = 100;
while 1
    y_pos = data(row, y_pos_column);
    %具体判断标准可以加上阈值
    if data(row, y_vel_column) ~= 0 && y_pos >= 68
        type = 1;
        break
    else
        row = row + 1;
    end
end

time_point = time + row;

y_pos = data(time_point, y_pos_column);
x_pos = data(time_point, x_pos_column);
x_pos_ego = data(time_point, ego_x_pos_column);
y_pos_ego = data(time_point, ego_y_pos_column);


longitudinal_distance = abs(x_pos - x_pos_ego);
lateral_distance = abs(y_pos - y_pos_ego);

yyaxis left
plot((time/100), lateral_distance, 'LineWidth', 1.5);
xlabel('Time(s)')
ylabel('Lateral Distance(m)')
ylim([10, 18]);

yyaxis right
plot((time/100), longitudinal_distance, 'LineWidth', 1.5);
ylabel('Longitudinal Distance(m)')
ylim([80, 160]);

% Change the font of the axis
set(gca, 'FontName', 'Times', 'FontSize', 12);





id = 17;
x_pos_column = (id - 1) * 8 + 10;
y_pos_column = (id - 1) * 8 + 11;
y_vel_column = (id - 1) * 8 + 13;

row = 100;
while 1
    x_pos = data(row, x_pos_column);
    y_pos = data(row, y_pos_column);
    y_vel = data(row, y_vel_column);
    ego_x_pos = data(row, ego_x_pos_column);
    if y_pos >= 56 && y_pos < 61 && y_vel <  -0.1
        type = 7;
        break
    else 
        row = row + 1;
    end
end

time_point = time + row;

y_pos = data(time_point, y_pos_column);
x_pos = data(time_point, x_pos_column);
x_pos_ego = data(time_point, ego_x_pos_column);
y_pos_ego = data(time_point, ego_y_pos_column);


longitudinal_distance = abs(x_pos - x_pos_ego);
lateral_distance = abs(y_pos - y_pos_ego);

figure;
yyaxis left
plot((time/100), lateral_distance, 'LineWidth', 1.5);
xlabel('Time(s)')
ylabel('Lateral Distance(m)')
ylim([0, 4]);

yyaxis right
plot((time/100), longitudinal_distance, 'LineWidth', 1.5);
ylabel('Longitudinal Distance(m)')
ylim([0, 80]);

% Change the font of the axis
set(gca, 'FontName', 'Times', 'FontSize', 12);

