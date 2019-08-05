clear; clc; close all;

dataset_dir = 'aorta_train_2/';

% volume_list = dir([dataset_dir '*_volume.mat']);
% label_list = dir([dataset_dir '*_label.mat']);
% 
% % data augmentation
% disp('------------------augmentation--------------------')
% for i = 1:numel(volume_list)
%     disp(i)
%     volume_file = volume_list(i);
%     volume_dir = [volume_file.folder '/' volume_file.name];
%     load(volume_dir)
%     % Intensity
%     volume = 1.5 * volume; volume(volume > 1) = 1;
%     save([volume_file.folder '/' num2str(i) '_intenh_volume.mat'], 'volume', '-v6')
%     % Rotation
% %     volume = permute(volume, [3,2,1]);
% %     volume2 = rot90(volume);
% %     volume = permute(volume2, [3,2,1]);
% %     save([volume_file.folder '/' num2str(i) '_rot90_volume.mat'], 'volume', '-v6')
% %     volume3 = rot90(volume2);
% %     volume = permute(volume3, [3,2,1]);
% %     save([volume_file.folder '/' num2str(i) '_rot180_volume.mat'], 'volume', '-v6')
% %     volume4 = rot90(volume3);
% %     volume = permute(volume4, [3,2,1]);
% %     save([volume_file.folder '/' num2str(i) '_rot270_volume.mat'], 'volume', '-v6')
%     
%     label_file = label_list(i);
%     label_dir = [label_file.folder '/' label_file.name];
%     load(label_dir)
%     % Intensity
%     save([label_file.folder '/' num2str(i) '_intenh_label.mat'], 'label', '-v6')
%     % Rotation
% %     label = permute(label, [3,2,1,4]);
% %     label2 = rot90(label);
% %     label = permute(label2, [3,2,1,4]);
% %     save([label_file.folder '/' num2str(i) '_rot90_label.mat'], 'label', '-v6')
% %     label3 = rot90(label2);
% %     label = permute(label3, [3,2,1,4]);
% %     save([label_file.folder '/' num2str(i) '_rot180_label.mat'], 'label', '-v6')
% %     label4 = rot90(label3);
% %     label = permute(label4, [3,2,1,4]);
% %     save([label_file.folder '/' num2str(i) '_rot270_label.mat'], 'label', '-v6')
% end

volume_list = dir([dataset_dir '*_volume.mat']);
label_list = dir([dataset_dir '*_label.mat']);

% shuffle
new_indices = randperm(numel(volume_list));
disp('------------------shuffle&rename--------------------')
for i = 1:numel(volume_list)
    volume_file = volume_list(i);
    volume_dir_old = [volume_file.folder '/' volume_file.name];
    volume_dir_new = [volume_file.folder '/' num2str(new_indices(i)) '_volume.mat'];
    
    % rename
    movefile(volume_dir_old, volume_dir_new)
    
    label_file = label_list(i);
    label_dir_old = [label_file.folder '/' label_file.name];
    label_dir_new = [label_file.folder '/' num2str(new_indices(i)) '_label.mat'];
    
    % rename
    movefile(label_dir_old, label_dir_new)
end
