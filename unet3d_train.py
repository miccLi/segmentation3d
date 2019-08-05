import numpy as np
import scipy.io as sio
import os, re, random, time
import tensorflow as tf

from unet3d import unet3d


def tversky_loss(labels, logits, output_channels, alpha=0.5, beta=0.5):
    prob = tf.nn.softmax(logits=logits)
    p0 = prob
    p1 = 1 - prob
    g0 = labels
    g1 = 1 - labels

    num = tf.reduce_sum(p0 * g0, axis=(0, 1, 2, 3))
    den = num + alpha * tf.reduce_sum(p0 * g1, axis=(0, 1, 2, 3)) + beta * tf.reduce_sum(p1 * g0, axis=(0, 1, 2, 3))

    t = tf.reduce_sum(num / den)

    return output_channels - t


if __name__ == '__main__':
    dataset = 'VISCERAL_multiple'  # 'VISCERAL_multiple', 'SLIVER07_liver'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cross_validation = 1
    restore = False
    vol_size = [8, 512, 512]

    if dataset == 'VISCERAL_multiple':
        data_train_path = '/data2/PEICHAO_LI/data/VISCERAL/multiple_2fold_%d_%d_%d/train_%d/' % (
            vol_size[2], vol_size[1], vol_size[0], cross_validation)
        data_test_path = '/data2/PEICHAO_LI/data/VISCERAL/multiple_2fold_%d_%d_%d/test_%d/' % (
            vol_size[2], vol_size[1], vol_size[0], cross_validation)
    elif dataset == 'SLIVER07_liver':  # TODO
        data_train_path = '/data2/XIAOYUN_ZHOU/SLiver07/liver_2fold/liver_train_%d/' % (cross_validation)
        data_test_path = '/data2/XIAOYUN_ZHOU/SLiver07/liver_2fold/liver_test_%d/' % (cross_validation)

    train_volume_list = sorted(
        [vol_name for vol_name in os.listdir(data_train_path) if vol_name.endswith('_volume.mat')])
    train_label_list = sorted(
        [label_name for label_name in os.listdir(data_train_path) if label_name.endswith('_label.mat')])
    test_volume_list = sorted([vol_name for vol_name in os.listdir(data_test_path) if vol_name.endswith('_volume.mat')])
    test_label_list = sorted(
        [label_name for label_name in os.listdir(data_test_path) if label_name.endswith('_label.mat')])
    train_num = len(train_volume_list)
    test_num = len(test_volume_list)

    batch_size = 1
    epoch = 6
    feature_channels = 16
    output_channels = 8
    downsampling = 3
    loss_type = 'cross_entropy'  # 'cross_entropy' or 'dice'

    step_show = 100

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    vol_in = tf.placeholder(dtype='float32', shape=[None, vol_size[0], vol_size[1], vol_size[2], 1])
    labels = tf.placeholder(dtype='float32', shape=[None, vol_size[0], vol_size[1], vol_size[2], output_channels])
    prediction, logits = unet3d(vol_in, labels, feature_channels=feature_channels, output_channels=output_channels,
                                downsampling=downsampling)
    if loss_type.upper() in ['DICE']:
        loss = tversky_loss(labels=tf.stop_gradient(labels), logits=logits, output_channels=output_channels)
    else:
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels), logits=logits))
    tf.summary.scalar('Loss', loss)
    # tf.summary.scalar('IoU', prediction['IoU'])

    for lr_train in [0.1, 0.05, 0.01, 0.005]:

        boundary = [train_num // batch_size, train_num * 2 // batch_size]
        lr_values = [lr_train, lr_train / 2, lr_train / 10]
        save_path = '/media/xz6214/4276F10376F0F90D/trained_model/%s/%d_%d_%d/unet3d/model_lr_%f_crossval_%s/' % (
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
        saver = tf.train.Saver(max_to_keep=epoch)
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
            for step in range((train_num // batch_size) * epoch):
                image_id_list = random.sample(range(train_num), batch_size)
                image_batch = []
                label_batch = []
                for image_id in image_id_list:
                    image_id = np.random.choice(train_num) + 1
                    image_np = sio.loadmat(data_train_path + "%s_volume.mat" % (image_id))['volume']
                    image_np = np.reshape(image_np, (1, image_np.shape[0], image_np.shape[1], image_np.shape[2], 1))
                    image_batch.append(image_np)
                    label_np = sio.loadmat(data_train_path + "%s_label.mat" % (image_id))['label']
                    label_np = np.reshape(label_np,
                                          (1, label_np.shape[0], label_np.shape[1], label_np.shape[2],
                                           label_np.shape[3]))
                    label_batch.append(label_np)
                image_batch = np.concatenate(image_batch, axis=0)
                label_batch = np.concatenate(label_batch, axis=0)

                # _, loss_value, summary, global_step_show, pred, lr_show = sess.run(
                #     [train, loss, merged_summary_op, global_step, prediction, lr],
                #     feed_dict={vol_in: image_batch, labels: label_batch})
                _, loss_value, summary, global_step_show, lr_show = sess.run(
                    [train, loss, merged_summary_op, global_step, lr],
                    feed_dict={vol_in: image_batch, labels: label_batch})

                if (step + 1) % step_show == 0:
                    total_loss += loss_value
                    total_loss = total_loss / step_show
                    # iou = pred['IoU']
                    print('Step: %d, Learning rate: %f, Loss: %f, Running time: %f' %
                          (global_step_show, lr_show, total_loss, time.time() - start_time))

                    total_loss = 0

                    writer.add_summary(summary, global_step=global_step_show)
                    writer.flush()
                    start_time = time.time()
                else:
                    total_loss += loss_value

                # Testing after each epoch
                if (step + 1) % (train_num // batch_size) == 0:
                    saved_path = saver.save(sess, save_path + 'Model', global_step=global_step)
                    print('-------------------------------------------------')
                    logfile.write('-------------------------------------------------\n')
                    i_patient = np.zeros([10, output_channels - 1])
                    u_patient = np.zeros([10, output_channels - 1])
                    for i in range(test_num):
                        volume_path = data_test_path + test_volume_list[i]
                        label_path = data_test_path + test_label_list[i]
                        assert volume_path[:volume_path.find('_volume.mat')] == label_path[
                                                                                :label_path.find('_label.mat')]

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
                        i_patient[patient_id - 1, :] += i_value
                        u_patient[patient_id - 1, :] += u_value

                    iou_all_patients = np.divide(i_patient, u_patient)
                    for patient_id in range(10):
                        msg = 'epoch %d, Testing IoU of each organ for patient %d: %s\n' % (
                            (step + 1) // (train_num // batch_size), patient_id,
                            ','.join(['%.3f' % n for n in iou_all_patients[patient_id, :]]))
                        print(msg)
                        logfile.write(msg)
                    msg = 'epoch %d, Current loss: %f, Average testing IoU of each organ for all %d patients: %s\n' % (
                        (step + 1) // (train_num // batch_size), loss_value, len(iou_all_patients),
                        ','.join(['%.3f' % n for n in np.mean(iou_all_patients, axis=0)]))
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
                        # sio.savemat(prediction_path + volume_name + '_prediction.mat',
                        #             {'probabilities': pred_dict['probabilities']})
                        # sio.savemat(prediction_path + volume_name + '_label.mat', {'label': label_np})
                    print('prediction complete.')
        logfile.close()
