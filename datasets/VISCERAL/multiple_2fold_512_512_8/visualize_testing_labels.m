clear; clc; close all;

volume_path = 'train_1/';
% volume_id = 'test_patient8_0_0_25';

volume_list = dir([volume_path '*_volume.mat']);
for volume_idx = 1:numel(volume_list)
    volume_file = volume_list(volume_idx);
    disp(volume_file)
    volume_id = volume_file.name(1:end-11);
    volume_mat = load([volume_path volume_id '_volume.mat']);
    volume = volume_mat.volume;
    label_mat = load([volume_path volume_id '_label.mat']);
    label = label_mat.label;

    volume = permute(volume, [3,2,1]);
    label = permute(label, [3,2,1,4]);
    background = label(:,:,:,1);
    liver = label(:,:,:,2);
    aorta= label(:,:,:,3);
    left_kidney = label(:,:,:,4);
    right_kidney = label(:,:,:,5);
    left_lung = label(:,:,:,6);
    right_lung = label(:,:,:,7);
    sternum = label(:,:,:,8);

    % Visualize each slice along with their labels
    for slice_index = 1:size(volume, 3)
        slice = mat2gray(volume(:,:,slice_index));
        imshow(slice, [])
        pause(0.2)

        slice_liver = liver(:,:,slice_index);
        slice_aorta = aorta(:,:,slice_index);
        slice_left_kidney = left_kidney(:,:,slice_index);
        slice_right_kidney = right_kidney(:,:,slice_index);
        slice_left_lung = left_lung(:,:,slice_index);
        slice_right_lung = right_lung(:,:,slice_index);
        slice_sternum = sternum(:,:,slice_index);

        slice_r = slice; slice_g = slice; slice_b = slice;
        % Blue: liver
        slice_b(slice_liver==1) = 1;
        % Red: aorta
        slice_r(slice_aorta==1) = 1;
        % Green: left_kidney
        slice_g(slice_left_kidney==1) = 1;
        % Magenta: right_kidney
        slice_r(slice_right_kidney==1) = 1;
        slice_b(slice_right_kidney==1) = 1;
        % Yellow: left_lung
        slice_r(slice_left_lung==1) = 1;
        slice_g(slice_left_lung==1) = 1;
        % Cyan: right_lung
        slice_b(slice_right_lung==1) = 1;
        slice_g(slice_right_lung==1) = 1;
        % White: sternum
        slice_r(slice_sternum==1) = 1;
        slice_g(slice_sternum==1) = 1;
        slice_b(slice_sternum==1) = 1;

        slice_labelled = cat(3, slice_r, slice_g, slice_b);

        imshow(slice_labelled, [])
        pause(0.3)
    end
end