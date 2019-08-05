import numpy as np
import scipy.io as sio
import os, re, random, time
import tensorflow as tf

from depthwise_dilated_unet import DDUNet

dataset = 'VISCERAL_aorta'  # 'VISCERAL_aorta', 'SLIVER07_liver'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
cross_validation = 1
restore = False
vol_size = [8, 512, 512]

if dataset == 'VISCERAL_aorta':
    data_train_path = '/data2/PEICHAO_LI/data/VISCERAL/aorta_2fold_augmented_%d_%d_%d/aorta_train_%d/' % (
        vol_size[2], vol_size[1], vol_size[0], cross_validation)
    data_test_path = '/data2/PEICHAO_LI/data/VISCERAL/aorta_2fold_augmented_%d_%d_%d/aorta_test_%d/' % (
        vol_size[2], vol_size[1], vol_size[0], cross_validation)
elif dataset == 'SLIVER07_liver':    # TODO
    data_train_path = '/data2/XIAOYUN_ZHOU/SLiver07/liver_2fold/liver_train_%d/' % (cross_validation)
    data_test_path = '/data2/XIAOYUN_ZHOU/SLiver07/liver_2fold/liver_test_%d/' % (cross_validation)

train_volume_list = sorted([vol_name for vol_name in os.listdir(data_train_path) if vol_name.endswith('_volume.mat')])
train_label_list = sorted([label_name for label_name in os.listdir(data_train_path) if label_name.endswith('_label.mat')])
test_volume_list = sorted([vol_name for vol_name in os.listdir(data_test_path) if vol_name.endswith('_volume.mat')])
test_label_list = sorted([label_name for label_name in os.listdir(data_test_path) if label_name.endswith('_label.mat')])
train_num = len(train_volume_list)
test_num = len(test_volume_list)

batch_size = 1
epoch = 12
feature_channels = 16
output_channels = 2
downsampling = 6
downsampling_type = 'conv'  # 'conv', 'max_pooling'
upsampling_type = 'bilinear'  # 'deconv', 'nearest_neighbour', 'bilinear'
norm_type = 'IN'            # 'IN', 'LN', 'BN'

step_show = 100

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

vol_in = tf.placeholder(dtype='float32', shape=[batch_size, vol_size[0], vol_size[1], vol_size[2], 1])
labels = tf.placeholder(dtype='float32', shape=[batch_size, vol_size[0], vol_size[1], vol_size[2], 2])
model = DDUNet(vol_in,
               labels,
               feature_channels,
               output_channels,
               downsampling,
               downsampling_type,
               upsampling_type,
               norm_type)
prediction = model.prediction
logits = model.logits
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels), logits=logits))
tf.summary.scalar('Loss', loss)
tf.summary.scalar('IoU', prediction['IoU'])

for lr_train in [0.1, 0.05, 0.01, 0.005]:

    boundary = [train_num//batch_size, train_num*4//batch_size, train_num*7//batch_size]
    lr_values = [lr_train, lr_train / 2, lr_train / 10, lr_train / 50]
    save_path = '/media/xz6214/4276F10376F0F90D/trained_model/%s/%d_%d_%d/ddunet/model_lr_%f_crossval_%s/' % (
        dataset, vol_size[2], vol_size[1], vol_size[0], lr_train, cross_validation)
    if not restore:
        os.system('rm -rf %s' % save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logfile = open(save_path + 'training_log.txt', 'w+')
    logfile.write('********************* LR = %f *********************\n' % lr_train)

    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.piecewise_constant(x=global_step, boundaries=boundary, values=lr_values)
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    train = optimizer.minimize(loss, global_step=global_step)
    tf.summary.scalar('Learning Rate', lr)

    merged_summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=3)
    with tf.Session(config=config) as sess:
        tf.train.write_graph(graph_or_graph_def=sess.graph_def, logdir=save_path, name='Model')
        writer = tf.summary.FileWriter(save_path, sess.graph)
        sess.run(init)
        total_loss = 0

        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print(total_parameters)
        logfile.write('total parameters: %d\n' % total_parameters)

        if restore:
            ckpt = tf.train.get_checkpoint_state(save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

        start_time = time.time()
        for step in range((train_num//batch_size)*epoch):
            image_id_list = random.sample(range(train_num), batch_size)
            image_batch = []
            label_batch = []
            for image_id in image_id_list:
                image_id = np.random.choice(train_num) + 1
                image_np = sio.loadmat(data_train_path + "%s_volume.mat" % (image_id))['volume']
                image_np = np.reshape(image_np, (1, image_np.shape[0], image_np.shape[1], image_np.shape[2], 1))
                image_batch.append(image_np)
                lable_np = sio.loadmat(data_train_path + "%s_label.mat" % (image_id))['label']
                lable_np = np.reshape(lable_np, (1, lable_np.shape[0], lable_np.shape[1], lable_np.shape[2], lable_np.shape[3]))
                label_batch.append(lable_np)
            image_batch = np.concatenate(image_batch, axis=0)
            label_batch = np.concatenate(label_batch, axis=0)

            _, loss_value, summary, global_step_show, pred, lr_show = sess.run([train, loss, merged_summary_op, global_step, prediction, lr], feed_dict={vol_in: image_batch, labels: label_batch})

            if (step+1) % step_show == 0:
                total_loss += loss_value
                total_loss = total_loss/step_show
                iou = pred['IoU']
                print('Step: %d, Learning rate: %f, Loss: %f, Running time: %f' %
                      (global_step_show, lr_show, total_loss, time.time() - start_time))

                total_loss = 0

                writer.add_summary(summary, global_step=global_step_show)
                writer.flush()
                start_time = time.time()
            else:
                total_loss += loss_value

            # Testing after each epoch
            if (step+1) % (train_num//batch_size) == 0:
                saved_path = saver.save(sess, save_path + 'Model', global_step=global_step)
                print('-------------------------------------------------')
                logfile.write('-------------------------------------------------\n')
                i_patient = np.zeros(10)
                u_patient = np.zeros(10)
                for i in range(len(test_volume_list)):
                    volume_path = data_test_path + test_volume_list[i]
                    label_path = data_test_path + test_label_list[i]
                    assert volume_path[:volume_path.find('_volume.mat')] == label_path[:label_path.find('_label.mat')]

                    volume_name = volume_path[:volume_path.find('_volume.mat')]
                    patient_id = volume_name[volume_name.find('test_patient') + 12:]
                    patient_id = int(patient_id[:patient_id.find('_')])

                    volume_np = sio.loadmat(volume_path)['volume']
                    volume_np = np.reshape(volume_np,
                                           (1, volume_np.shape[0], volume_np.shape[1], volume_np.shape[2], 1))

                    label_np = sio.loadmat(label_path)['label']
                    label_np = np.reshape(label_np,
                                          (1, label_np.shape[0], label_np.shape[1], label_np.shape[2],
                                           label_np.shape[3]))

                    [pred, ] = sess.run([prediction], feed_dict={vol_in: volume_np, labels: label_np})
                    i_value = pred['And']
                    u_value = pred['Or']

                    # Calculate IoU for each patient
                    i_patient[patient_id-1] += np.sum(i_value)
                    u_patient[patient_id-1] += np.sum(u_value)

                iou_list = np.divide(i_patient, u_patient)
                assert len(iou_list) == 10
                for patient_id in range(10):
                    msg = 'epoch %d, Testing IoU for patient %d: %f\n' % ((step+1) // (train_num//batch_size), patient_id, iou_list[patient_id])
                    print(msg)
                    logfile.write(msg)
                msg = 'epoch %d, Current loss: %f, Average testing IoU for all %d patients: %f\n' % ((step+1) // (train_num//batch_size), loss_value, len(iou_list), np.mean(iou_list))
                print(msg)
                logfile.write(msg)
                print('-------------------------------------------------')
                logfile.write('-------------------------------------------------\n\n')
                logfile.flush()
                start_time = time.time()

            # Save prediction after each epoch
            if (step + 1) % (train_num // batch_size) == 0:
                for i in range(len(test_volume_list)):
                    volume_path = data_test_path + test_volume_list[i]
                    label_path = data_test_path + test_label_list[i]
                    assert volume_path[:volume_path.find('_volume.mat')] == label_path[
                                                                            :label_path.find('_label.mat')]

                    volume_name = volume_path[:volume_path.find('_volume.mat')]
                    volume_name = volume_name[volume_name.rfind('/') + 1:]
                    indices = re.search('test_patient[0-9]+', volume_name).span()
                    patient_name = volume_name[indices[0]:indices[1]]

                    volume_np = sio.loadmat(volume_path)['volume']
                    volume_np = np.reshape(volume_np,
                                           (1, volume_np.shape[0], volume_np.shape[1], volume_np.shape[2], 1))

                    label_np = sio.loadmat(label_path)['label']
                    label_np = np.reshape(label_np,
                                          (1, label_np.shape[0], label_np.shape[1], label_np.shape[2],
                                           label_np.shape[3]))

                    [pred_dict, ] = sess.run([prediction], feed_dict={vol_in: volume_np, labels: label_np})
                    prediction_path = save_path + 'prediction_' + str(
                        (step + 1) // (train_num // batch_size)) + '/' + patient_name + '/'
                    if not os.path.exists(prediction_path):
                        os.makedirs(prediction_path)
                    sio.savemat(prediction_path + volume_name + '_prediction.mat',
                                {'probabilities': pred_dict['probabilities']})
                    sio.savemat(prediction_path + volume_name + '_label.mat', {'label': label_np})
                print('prediction complete.')
    logfile.close()
