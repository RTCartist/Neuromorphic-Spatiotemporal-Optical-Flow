% Main execution
base_folder = '../data/uav';
cutgreystore = fullfile(base_folder, 'GREY');
compressed_image_path = fullfile(base_folder, 'Compressed Image');
cropped_image_path = fullfile(base_folder, 'Cropped Image');
differ_matrix_path = fullfile(base_folder, 'differ_matrix');
value_matrix_path = fullfile(base_folder, 'value_matrix');

% construct folder
if ~exist(compressed_image_path, 'dir')
    mkdir(compressed_image_path);
end

if ~exist(cropped_image_path, 'dir')
    mkdir(cropped_image_path);
end

if ~exist(differ_matrix_path, 'dir')
    mkdir(differ_matrix_path);
end

if ~exist(value_matrix_path, 'dir')
    mkdir(value_matrix_path);
end

params = struct('alphaoff', 1, 'alphaon', 1, ...
    'voff', -0.2, 'von', 0.1, ...
    'koff', 51.03, 'kon', -2.91, ...
    'son', 0.2, 'soff', 0.8, ...
    'bon', -5.12, 'boff', 3.10, ...
    'Ron', 163305, 'Roff', 2104377, ...
    'won', 1, 'woff', 0, ...
    'wini', 0.5);

% drain-source voltage
vds = 1;

m = 40;
n = 40;
th1 = 0.7;
th2 = 1.5;


% Set the range and interval for reading images
start_idx = 1;
end_idx = 32;
interval = 1;

% Set processing region (coordinates of the upper left and lower right corners)
region_ul = [276, 879]; % [y, x] of upper left corner
region_lr = [436, 1039]; % [y, x] of lower right corner

compressed_images = process_images(cutgreystore, m, n, compressed_image_path, cropped_image_path, start_idx, end_idx, interval, region_ul, region_lr);

dt = 0.0005;
% precise simulation
nSubSteps = 1000;
% fast simulation
% nSubSteps = 1;

tPipelineStart = tic; % START MASTER TIME
[w_array, resistances_over_time, diff_matrices, value_matrices, frameTimes] = simulate_memristor_array(compressed_images, dt, nSubSteps, th1, th2, params);
pipelineTime = toc(tPipelineStart);   % << END master timer

save(fullfile(base_folder, 'w_array.mat'), 'w_array');
save(fullfile(base_folder, 'resistances_over_time.mat'), 'resistances_over_time');
save(fullfile(base_folder, 'diff_matrices.mat'), 'diff_matrices'); % Save difference matrices
save(fullfile(base_folder, 'value_matrices.mat'), 'value_matrices');
save(fullfile(base_folder, 'frameTimes.mat'), 'frameTimes');

resistances_matrix_path = fullfile(base_folder, 'resistances_matrix'); % New folder path
if ~exist(resistances_matrix_path, 'dir')
    mkdir(resistances_matrix_path);
end

%%
% save_value_matrices2(value_matrices, value_matrix_path);
% save_resistances_matrices(resistances_over_time, resistances_matrix_path);
% Visualize difference matrices and save to folder
% save_difference_matrices(diff_matrices, differ_matrix_path);
%% 

% Image processing
function images = read_images_from_folder(folder_path, start_idx, end_idx, interval)
    files = dir(fullfile(folder_path, '*.jpg'));
    filenames = {files.name};
    
    % Extract numeric part from filenames
    numeric_part = cellfun(@(x) sscanf(x, '%d'), filenames);
    
    % Sort numerically
    [~, sorted_indices] = sort(numeric_part);
    sorted_filenames = filenames(sorted_indices);
    
    % Apply range and interval
    selected_indices = start_idx:interval:min(end_idx, length(sorted_filenames));
    sorted_filenames = sorted_filenames(selected_indices);
    
    images = cell(size(sorted_filenames));
    for i = 1:length(sorted_filenames)
        images{i} = imread(fullfile(folder_path, sorted_filenames{i}));
    end
end

% Crop image to processing region
function cropped_image = crop_image(image, region_ul, region_lr)
    cropped_image = image(region_ul(1):region_lr(1), region_ul(2):region_lr(2), :);
end

% Compress image with higher precision
function compressed_image = compress_image(image, m, n)
    [height, width, ~] = size(image);
    new_height = floor(height / n);
    new_width = floor(width / m);
    
    % Convert image to double precision for higher precision during resizing
    image_double = im2double(image);
    
    % Resize image using Lanczos-3 interpolation
    compressed_image = imresize(image_double, [new_height, new_width], 'lanczos3');
end

% Save image
function save_image(image, path)
    imwrite(image, path);
end

% Process images: crop, compress, and save
function compressed_images = process_images(folder_path, m, n, compressed_image_path, cropped_image_path, start_idx, end_idx, interval, region_ul, region_lr)
    images = read_images_from_folder(folder_path, start_idx, end_idx, interval);
    compressed_images = cell(size(images));

    if ~exist(compressed_image_path, 'dir')
        mkdir(compressed_image_path);
    end

    for i = 1:length(images)
        cropped_image = crop_image(images{i}, region_ul, region_lr);
        save_image(cropped_image, fullfile(cropped_image_path, sprintf('cropped_%d.jpg', i)));
        compressed_images{i} = compress_image(cropped_image, m, n);
        save_image(compressed_images{i}, fullfile(compressed_image_path, sprintf('compressed_%d.jpg', i)));
    end
end

% Calculate difference matrix between two image arrays
function result = calculate_difference_matrix(image1, image2, th1, th2, func1, func2, func3)
    data1 = double(image1);
    data2 = double(image2);
    diff = abs(data1 - data2);
    
    result = zeros(size(diff));
    result(diff > th2) = func3(diff(diff > th2));
    result(diff > th1 & diff <= th2) = func2(diff(diff > th1 & diff <= th2));
    result(diff <= th1) = func1(diff(diff <= th1));
end

% Custom functions greater than threshold function1
function out = func1(x)
% 0.2*threshold
    out = (x - 5.5) * 0.6;
end

function out = func2(x)
% 1.6*threshold
    out = (x + 4) * 0.75;
end

function out = func3(x)
%   out = (1.2 - x) * 0.75;
    out = (x + 4) * 0.75;
end

function new_w = update_state(w, V, dt, params)
    if V < params.voff
        dwdt = params.koff * (V / params.voff - 1) ^ params.alphaoff * (1-w*params.soff)^ params.boff;
    elseif V > params.von
        dwdt = params.kon * (V / params.von - 1) ^ params.alphaon * (1-w*params.son)^ params.bon;
    else
        dwdt = 0; % If V does not meet any of the above conditions, set dwdt to 0
    end

    new_w = w + dwdt * dt;
    new_w = max(0, min(new_w, 1)); % Use window function to limit the value of w
end

% Simulate memristor array
function [w_array, resistances_over_time, diff_matrices, value_matrices, frameTimes] = simulate_memristor_array(compressed_images, dt, nSubSteps, th1, th2, params)
    
    dt_sub = dt / nSubSteps;
    
    [height, width, ~] = size(compressed_images{1});
    w_array = params.wini * ones(height, width);
    resistances_over_time = {};
    % add this line to store the initial value
    resistances = calculate_resistances_exp(w_array, params);
    resistances_over_time{end + 1} = resistances;

    diff_matrices = {}; % Initialize cell array for difference matrices
    value_matrices = {}; % Initialize value matrices
    frameTimes = zeros(1, length(compressed_images)-1);

    for i = 1:length(compressed_images) - 1

        diff_matrix = calculate_difference_matrix(double(compressed_images{i})*256, double(compressed_images{i + 1})*256, th1, th2, @func1, @func2, @func3);
        diff_matrices{end + 1} = diff_matrix; %#ok<AGROW>

        value_matrix = abs(double(compressed_images{i + 1}) - double(compressed_images{i}));
        value_matrix = value_matrix*256;
        value_matrices{end + 1} = value_matrix; %#ok<AGROW>
        
        tFrame = tic;
        for y = 1:height
            for x = 1:width
                V = diff_matrix(y, x);
                v_mod = modulatefunc (V);
                % w_array(y, x) = update_state(w_array(y, x), v_mod, dt, params);
                for s = 1:nSubSteps
                    w_array(y, x) = update_state(w_array(y, x), v_mod, dt_sub, params);
                end  
            end
        end

        resistances = calculate_resistances_exp(w_array, params);
        resistances_over_time{end + 1} = resistances; %#ok<AGROW>
        frameTimes(i) = toc(tFrame);
    end
end

function resistances = calculate_resistances_linear(w_array, params)
    resistances = params.Ron + (params.Roff - params.Ron) * w_array;
end

function resistances = calculate_resistances_exp(w_array, params)
    lambda = reallog(params.Roff / params.Ron);
    resistances = params.Ron ./ exp(-lambda * (1 - w_array));
end

% Visualize difference matrix
function save_difference_matrices(diff_matrices, differ_matrix_path)
    num_matrices = length(diff_matrices);
    for i = 1:num_matrices
        % Create a new image and save it as a file
        img = imagesc(diff_matrices{i}); % Use imagesc to display the difference matrix
        colorbar; % Add color bar
        caxis([-10 10]); % Set the color bar range from 2k to 10k
        title(sprintf('Difference Matrix %d', i));
        xlabel('Width');
        ylabel('Height');
        axis image; % Keep the aspect ratio of the image
        % Display the values on the heatmap
        [rows, cols] = size(diff_matrices{i});
        for r = 1:rows
            for c = 1:cols
                text(c, r, num2str(diff_matrices{i}(r, c), '%.2f'), ...
                    'HorizontalAlignment', 'center', 'Color', 'k');
            end
        end
        saveas(gcf, fullfile(differ_matrix_path, sprintf('difference_matrix_%d.png', i)));
        close; % Close the image window
    end
end

function save_value_matrices(value_matrices,value_matrix_path)
    num_matrices = length(value_matrices);
    for i = 1:num_matrices
        % Create a new image and save it as a file
        figure;
        imagesc(value_matrices{i}); % Use imagesc to display the difference matrix
        colorbar; % Add color bar
        title(sprintf('Value Matrix %d', i));
        xlabel('Width');
        ylabel('Height');
        axis image; % Keep the aspect ratio of the image
        
        % % Display the values on the heatmap with higher precision
        % [rows, cols] = size(value_matrices{i});
        % for r = 1:rows
        %     for c = 1:cols
        %         text(c, r, num2str(value_matrices{i}(r, c), '%.4f'), ...
        %             'HorizontalAlignment', 'center', 'Color', 'k');
        %     end
        % end
        
        saveas(gcf, fullfile(value_matrix_path, sprintf('value_matrix_%d.png', i)));
        close; % Close the image window
    end
end

function save_value_matrices2(value_matrices, value_matrix_path)
    num_matrices = length(value_matrices);
    for i = 1:num_matrices
        % Create a new image and save it as a file
        figure('units','normalized','outerposition',[0 0 1 1]); % Full screen display
        imagesc(value_matrices{i}); % Use imagesc to display the difference matrix
        colorbar; % Add color bar
        title(sprintf('Value Matrix %d', i), 'FontSize', 16);
        xlabel('Width', 'FontSize', 14);
        ylabel('Height', 'FontSize', 14);
        axis image; % Keep the aspect ratio of the image
        
        % Display the values on the heatmap with higher precision
        [rows, cols] = size(value_matrices{i});
        for r = 1:rows
            for c = 1:cols
                text(c, r, num2str(value_matrices{i}(r, c), '%.2f'), ...
                    'HorizontalAlignment', 'center', 'Color', 'k', 'FontSize', 7);
            end
        end
        
        % Save image
        saveas(gcf, fullfile(value_matrix_path, sprintf('value_matrix_%d.png', i)));
        close; % Close the image window
    end
end


function save_resistances_matrices(resistances_over_time, resistances_path)
    num_matrices = length(resistances_over_time);
    for i = 1:num_matrices
        img = imagesc(resistances_over_time{i}); % Use imagesc to display the resistance matrix
        colorbar; % Add color bar
        caxis([160000 2104000]); % Set the color bar range from 2k to 10k
        title(sprintf('Resistance Matrix %d', i));
        xlabel('Width');
        ylabel('Height');
        axis image; % Keep the aspect ratio of the image
        saveas(gcf, fullfile(resistances_path, sprintf('resistance_matrix_%d.png', i)));
        close; % Close the image window
    end
end

function v_mod = modulatefunc (V)
% a = 0.7;
% b = 4.5;
% c = 1.7;
% d = 0;
a = 0.3;
b = 0;
c = 3;
d = -3;
if (V > 0)
    v_mod = a * V + b;
elseif (V < 0)
    v_mod = c * V + d;
end
v_mod = -v_mod;
end