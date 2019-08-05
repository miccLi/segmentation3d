clear; clc; close all;

patient_num = 10;
prediction_folder = '/data/PEICHAO_LI/liver/Unet_IN/model_lr_0.005000_crossval_1/prediction_12/';

result = zeros(2, patient_num);
fid = fopen([prediction_folder 'result.txt'], 'w');
for i = 1:patient_num
    patient_folder = [prediction_folder 'test_patient' num2str(i) '/'];
    prediction_list = dir([patient_folder '*_prediction.mat']);
    label_list = dir([patient_folder '*_label.mat']);
    
    and_list = zeros(1); or_list = zeros(1);
    fg_list = zeros(1); fg_gt_list = zeros(1);
    for j = 1:numel(prediction_list)
        % Prediction
        load([prediction_list(j).folder '/' prediction_list(j).name])
        % Label
        load([label_list(j).folder '/' label_list(j).name])
        
        pred = (classes>=0.5);
        foreground = pred(:,:,:,:,2);
        label = label>0.5;
        foreground_gt = label(:,:,:,:,2);
        fg_list(j) = sum(foreground(:));
        fg_gt_list(j) = sum(foreground_gt(:));
        
        and = foreground&foreground_gt;
        and_list(j) = sum(and(:));
        or = foreground|foreground_gt;
        or_list(j) = sum(or(:));
    end
    iou = sum(and_list) / sum(or_list);
    dice = (2*sum(and_list)) / (sum(fg_list)+sum(fg_gt_list));
    sprintf('patient %d, IoU is %f, Dice is %f', i, iou, dice)
    result(1, i) = iou; result(2, i) = dice;
end
mean(result(1,:))
mean(result(2,:))
fprintf(fid, 'Average IoU is %f, Standard Deviation is %f\n', mean(result(1,:)), std(result(1,:)));
fprintf(fid, 'Average Dice is %f, Standard Deviation is %f\n', mean(result(2,:)), std(result(2,:)));
fclose(fid);