clear; clc; close all;

path_vol = 'E:\datasets\VISCERAL\original\CT\';
path_seg = 'E:\datasets\VISCERAL\original\CTSeg\';

patient_num = 20;
patients = load(strcat(path_vol, 'Patient_id.txt'));
anatomies = load(strcat(path_seg, 'Anatomy_id.txt'));

% List of target anatomies and their values to be assigned:
% liver (index = 1, RadLexID = 58) = 1
% aorta (index = 4, RadLexID = 480) = 2
% left kidney (index = 11, RadLexID = 29663) = 3
% right kidney (index = 10, RadLexID = 29662) = 4
% left lung (index = 7, RadLexID = 1326) = 5
% right lung (index = 6, RadLexID = 1302) = 6
% sternum (index = 8, RadLexID = 2473) = 7
target_anatomy_index = [1,4,11,10,7,6,8];

boundary_z = zeros(patient_num, 2);
boundary_z(:,2) = 1000;

load('index_ref/indices.mat')   % load indices

patch_size = [512, 512, 8];
train_stride = [1, 1, 1];
test_stride = [1, 1, 8];
padding = [0, 0, patch_size(3)/2];

for patient_index = 1:patient_num
    fprintf('Patient No. %d\n', patient_index)
    patient_id = patients(patient_index, 1);
    image_name = strcat(path_vol, sprintf('100000%02d_1_CT_wb.nii', patient_id));
    volume = niftiread(image_name);
    
    % Find lower and upper boundaries to crop
    for anatomy_index = target_anatomy_index
        fprintf('\tFinding boundaries for anatomy No. %d\n', anatomy_index)
        anatomy_id = anatomies(anatomy_index, 1);
        label_name = strcat(path_seg, sprintf('100000%02d_1_CT_wb_%d.nii', patient_id, anatomy_id));
        segmentation = niftiread(label_name);
        
        % Lower boundary
        for i = 1:size(volume,3)
            label_slice = reshape(segmentation(:,:,i), size(segmentation,1), size(segmentation,2));
            tmp = find(label_slice);
            if size(tmp) > 0
                if boundary_z(patient_index,1) == 0
                    boundary_z(patient_index,1) = i;
                else
                    boundary_z(patient_index,1) = min(boundary_z(patient_index,1),i);
                end
                break
            end
        end
        
        % Upper boundary
        for i = size(volume,3):-1:1
            label_slice = reshape(segmentation(:,:,i), size(segmentation,1), size(segmentation,2));
            tmp = find(label_slice);
            if size(tmp) > 0
                if boundary_z(patient_index,2) == 1000
                    boundary_z(patient_index,2) = i;
                else
                    boundary_z(patient_index,2) = max(boundary_z(patient_index,2),i);
                end
                break
            end
        end
        
        % Make the cropped volume size divisible by patch size
        remainder = mod(boundary_z(patient_index,2)-boundary_z(patient_index,1), patch_size(3)/2);
        boundary_z(patient_index,2) = boundary_z(patient_index,2) + ceil(remainder/2);
        boundary_z(patient_index,1) = boundary_z(patient_index,1) - floor(remainder/2);
    end
    
    % Generate the volume
    volume_crop = volume(:,:,boundary_z(patient_index,1)+patch_size(3)/2:boundary_z(patient_index,2)-patch_size(3)/2);
    
    % Generate the label
    label = zeros([size(volume_crop), numel(target_anatomy_index)+1]);
    for anatomy_index = target_anatomy_index
        fprintf('\tGenerating labels for anatomy No. %d\n', anatomy_index)
        label_each = zeros(size(volume_crop));
        anatomy_id = anatomies(anatomy_index, 1);
        label_name = strcat(path_seg, sprintf('100000%02d_1_CT_wb_%d.nii', patient_id, anatomy_id));
        segmentation = niftiread(label_name);
        segmentation_crop = segmentation(:,:,boundary_z(patient_index,1)+patch_size(3)/2:boundary_z(patient_index,2)-patch_size(3)/2);
        
        label_each(segmentation_crop > 0) = 1;
        label(:,:,:,find(target_anatomy_index==anatomy_index,1)+1) = single(label_each);
    end
    label(:,:,:,1) = 1-any(label(:,:,:,2:end),4);
    
    % Save
%     fprintf('\tSaving data ... ')
%     save([num2str(patient_index) '.mat'], 'volume_crop');
%     save([num2str(patient_index) '_label.mat'], 'label');
%     fprintf('complete\n')
    
    V = single(volume_crop);
    V_cropped = (V - min(V(:))) / (max(V(:))-min(V(:)));
    L_cropped = single(label);
    vol_size = size(V_cropped);
    
    % Generate volumes
    fprintf('\tGenerating volumes ... \n')
    [flag, new_index] = find(indices==patient_index);
    count = 1;
    for mode = 0:1  % 0:train, 1:test
        if mode==0
            stride = train_stride;
            save_path = ['aorta_train_' num2str(flag) '/'];
            if ~exist(save_path, 'dir')
                mkdir(save_path);
            end
        else
            stride = test_stride;
            save_path = ['aorta_test_' num2str(3-flag) '/'];
            if ~exist(save_path, 'dir')
                mkdir(save_path);
            end
        end

        for j = 1:(vol_size(1)-patch_size(1))/stride(1)+1
            for k = 1:(vol_size(2)-patch_size(2))/stride(2)+1
                for l = 1:(vol_size(3)-patch_size(3))/stride(3)+1
                    % Crop to get training/testing data
                    volume = V_cropped(:, ...
                                       :, ...
                                       (l-1)*stride(3)+1:(l-1)*stride(3)+patch_size(3)  ...
                    );
                    label  = L_cropped(:, ...
                                       :, ...
                                       (l-1)*stride(3)+1:(l-1)*stride(3)+patch_size(3), ...
                                       :  ...
                    );
                    
                    volume = permute(volume, [3,2,1]);
                    label = permute(label, [3,2,1,4]);
                    
                    % save
                    if mode == 0
                        save([save_path num2str(new_index) '_' num2str(count) '_volume.mat'], 'volume', '-v6')
                        save([save_path num2str(new_index) '_' num2str(count) '_label.mat'], 'label', '-v6')
                    else
                        save([save_path 'test_patient' num2str(new_index) sprintf('_%d_%d_%d',j-1,k-1,l-1) '_volume.mat'], 'volume', '-v6')
                        save([save_path 'test_patient' num2str(new_index) sprintf('_%d_%d_%d',j-1,k-1,l-1) '_label.mat'], 'label', '-v6')
                    end
                    count = count + 1;

                    fprintf('==> Patient %d, volume_size [%d,%d,%d], label_size [%d,%d,%d,%d], patch [%d,%d,%d]\n', ...
                        patient_index, size(volume), size(label), j-1,k-1,l-1)
                end
            end
        end
        count = 1;
    end
    fprintf('\n')
end