from __future__ import print_function

# Import MNIST data
import sys
import getopt
import input_data
import os.path
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

class Usage(Exception):
    def __init__ (self,msg):
        self.msg = msg

# Parameters
learning_rate = 1e-4
training_epochs = 12
batch_size = 128
display_step = 1

# Network Parameters
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

n_hidden_1 = 300# 1st layer number of features
n_hidden_2 = 100# 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.5

'''
pruning Parameters
'''
# sets the threshold
prune_threshold_cov = 0.08
prune_threshold_fc = 1
# Frequency in terms of number of training iterations
prune_freq = 100
ENABLE_PRUNING = 0

def compute_file_name(p):
    name = ''
    name += 'cov' + str(int(p['cov1'] * 10))
    name += 'cov' + str(int(p['cov2'] * 10))
    name += 'fc' + str(int(p['fc1'] * 10))
    name += 'fc' + str(int(p['fc2'] * 10))
    return name

# Store layers weight & bias
def initialize_variables(parent_dir, model_number):
    with open(parent_dir + 'weight_crate' + model_number +'.pkl','rb') as f:
        wc1, wc2, wd1, out, bc1, bc2, bd1, bout = pickle.load(f)
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'cov1': tf.Variable(wc1),
        # 5x5 conv, 32 inputs, 64 outputs
        'cov2': tf.Variable(wc2),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'fc1': tf.Variable(wd1),
        # 1024 inputs, 10 outputs (class prediction)
        'fc2': tf.Variable(out)
    }

    biases = {
        'cov1': tf.Variable(bc1),
        'cov2': tf.Variable(bc2),
        'fc1': tf.Variable(bd1),
        'fc2': tf.Variable(bout)
    }
    return (weights, biases)
# weights = {
#     'cov1': tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1)),
#     'cov2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)),
#     'fc1': tf.Variable(tf.random_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 1024])),
#     'fc2': tf.Variable(tf.random_normal([1024, NUM_LABELS]))
# }
# biases = {
#     'cov1': tf.Variable(tf.random_normal([32])),
#     'cov2': tf.Variable(tf.random_normal([64])),
#     'fc1': tf.Variable(tf.random_normal([1024])),
#     'fc2': tf.Variable(tf.random_normal([10]))
# }
#
#store the masks
# weights_mask = {
#     'cov1': tf.Variable(tf.ones([5, 5, NUM_CHANNELS, 32]), trainable = False),
#     'cov2': tf.Variable(tf.ones([5, 5, 32, 64]), trainable = False),
#     'fc1': tf.Variable(tf.ones([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512]), trainable = False),
#     'fc2': tf.Variable(tf.ones([512, NUM_LABELS]), trainable = False)
# }
    # else:
    #     with open('assets.pkl','rb') as f:
    #         (weights, biases, weights_mask) = pickle.load(f)

# weights_mask = {
#     'cov1': np.ones([5, 5, NUM_CHANNELS, 32]),
#     'cov2': np.ones([5, 5, 32, 64]),
#     'fc1': np.ones([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512]),
#     'fc2': np.ones([512, NUM_LABELS])
# }
# Create model
def conv_network(x, weights, biases, keep_prob):
    conv = tf.nn.conv2d(x,
                        weights['cov1'],
                        strides = [1,1,1,1],
                        padding = 'VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['cov1']))
    pool = tf.nn.max_pool(
            relu,
            ksize = [1 ,2 ,2 ,1],
            strides = [1, 2, 2, 1],
            padding = 'VALID')

    conv = tf.nn.conv2d(pool,
                        weights['cov2'],
                        strides = [1,1,1,1],
                        padding = 'VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['cov2']))
    pool = tf.nn.max_pool(
            relu,
            ksize = [1 ,2 ,2 ,1],
            strides = [1, 2, 2, 1],
            padding = 'VALID')
    '''get pool shape'''
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [-1, pool_shape[1]*pool_shape[2]*pool_shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, weights['fc1']) + biases['fc1'])
    hidden = tf.nn.dropout(hidden, keep_prob)
    output = tf.matmul(hidden, weights['fc2']) + biases['fc2']
    return output , reshape

def calculate_non_zero_weights(weight):
    count = (weight != 0).sum()
    size = len(weight.flatten())
    return (count,size)

'''
Prune weights, weights that has absolute value lower than the
threshold is set to 0
'''
def dynamic_surgery(weight, pruning_th, recover_percent):
    threshold = np.percentile(np.abs(weight),pruning_th)
    weight_mask = np.abs(weight) > threshold
    tmp = (pruning_th) / float(100) * recover_percent
    soft_weight_mask = (1 - weight_mask) * (np.random.rand(*weight.shape) > (1-tmp))
    return (weight_mask, soft_weight_mask)

def prune_weights(weights, biases, org_masks, cRates, iter_cnt, parent_dir):
    keys = ['cov1','cov2','fc1','fc2']
    new_mask = {}
    for key in keys:
        w_eval = weights[key].eval()
        threshold_off = 0.9*(np.mean(w_eval) + cRates[key] * np.std(w_eval))
        threshold_on = 1.1*(np.mean(w_eval) + cRates[key] * np.std(w_eval))
        # elements at this postion becomes zeros
        mask_off = np.abs(w_eval) < threshold_off
        # elements at this postion becomes ones
        mask_on = np.abs(w_eval) > threshold_on
        new_mask[key] = np.logical_or(((1 - mask_off) * org_masks[key]),mask_on).astype(int)
    file_name_part = compute_file_name(cRates)
    mask_file_name = parent_dir+'mask_crate'+ file_name_part+'.pkl'
    file_name = parent_dir+'weight_crate'+ file_name_part+'.pkl'
    print("training done, save a mask file at "  + mask_file_name)
    with open(mask_file_name, 'wb') as f:
        pickle.dump(new_mask, f)
    mask_info(new_mask)
    print("Pruning done, dorp weights to {}".format(file_name))
    with open(file_name, 'wb') as f:
        pickle.dump((
            weights['cov1'].eval(),
            weights['cov2'].eval(),
            weights['fc1'].eval(),
            weights['fc2'].eval(),
            biases['cov1'].eval(),
            biases['cov2'].eval(),
            biases['fc1'].eval(),
            biases['fc2'].eval()),f)

'''
mask gradients, for weights that are pruned, stop its backprop
'''
def mask_gradients(weights, grads_and_names, weight_masks, biases):
    new_grads = []
    keys = ['cov1','cov2','fc1','fc2']
    for grad, var_name in grads_and_names:
        # flag set if found a match
        flag = 0
        index = 0
        for key in keys:
            if (weights[key]== var_name):
                # print(key, weights[key].name, var_name)
                mask = weight_masks[key]
                new_grads.append((tf.multiply(tf.constant(mask, dtype = tf.float32),grad),var_name))
                flag = 1
            # if (biases[key] == var_name):
            #     mask = biases_mask[key]
            #     new_grads.append((tf.multiply(tf.constant(mask, dtype = tf.float32),grad),var_name))
            #     flag = 1
        # if flag is not set
        if (flag == 0):
            new_grads.append((grad,var_name))
        # print(grad.get_shape())
    return new_grads

'''
plot weights and store the fig
'''
def plot_weights(weights,pruning_info):
        keys = ['cov1','cov2','fc1','fc2']
        fig, axrr = plt.subplots( 2, 2)  # create figure &  axis
        fig_pos = [(0,0), (0,1), (1,0), (1,1)]
        index = 0
        for key in keys:
            weight = weights[key].eval().flatten()
            # print (weight)
            size_weight = len(weight)
            weight = weight.reshape(-1,size_weight)[:,0:size_weight]
            x_pos, y_pos = fig_pos[index]
            #take out zeros
            weight = weight[weight != 0]
            # print (weight)
            hist,bins = np.histogram(weight, bins=100)
            width = 0.7 * (bins[1] - bins[0])
            center = (bins[:-1] + bins[1:]) / 2
            axrr[x_pos, y_pos].bar(center, hist, align = 'center', width = width)
            axrr[x_pos, y_pos].set_title(key)
            index = index + 1
        fig.savefig('fig_v3/weights'+pruning_info)
        plt.close(fig)

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1, 1)

def recover_weights(weights_mask, biases_mask, soft_weight_mask, soft_biase_mask):
    keys = ['cov1','cov2','fc1','fc2']
    mask_info(weights_mask)
    prev = weights_mask['fc1']
    for key in keys:
        weights_mask[key] = weights_mask[key] + (soft_weight_mask[key] * np.random.rand(*soft_weight_mask[key].shape))
    print("test in recover weights")
    print(np.array_equal(prev, weights_mask['fc1']))
    mask_info(weights_mask)
    return (weights_mask, biases_mask)
'''
Define a training strategy
'''
def main(argv = None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            # opts, args = getopt.getopt(argv[1:],'hp:tc1:tc2:tfc1:tfc2:')
            opts = argv
            threshold = {
                'cov1' : 0.08,
                'cov2' : 0.08,
                'fc1' : 1,
                'fc2' : 1
            }
            PRUNE_ONLY = False
            TRAIN = False
            SAVE = False
            for item in opts:
                print (item)
                opt = item[0]
                val = item[1]
                if (opt == '-m'):
                    model_number = val
                if (opt == '-learning_rate'):
                    learning_rate = val
                if (opt == '-prune'):
                    PRUNE_ONLY = val
                if (opt == '-train'):
                    TRAIN = val
                if (opt == '-parent_dir'):
                    parent_dir = val
                if (opt == '-recover_rate'):
                    recover_rates = val
                if (opt == '-crate'):
                    crate = val
                if (opt == '-iter_cnt'):
                    iter_cnt = val
                if (opt == '-next_iter_save'):
                    SAVE = val
            print('Train values:',TRAIN)
        except getopt.error, msg:
            raise Usage(msg)

        file_name_part = compute_file_name(crate)
        if (SAVE):
            mask_file = parent_dir +  'mask_crate'+ model_number +'.pkl'
        else:
            mask_file = parent_dir +  'mask_crate'+ file_name_part +'.pkl'

        if (TRAIN == True or SAVE == True):
            with open(mask_file,'rb') as f:
                weights_mask = pickle.load(f)
        else:
            weights_mask = {
                'cov1': np.ones([5, 5, NUM_CHANNELS, 20]),
                'cov2': np.ones([5, 5, 20, 50]),
                'fc1': np.ones([4 * 4 * 50, 500]),
                'fc2': np.ones([500, NUM_LABELS])
            }
            biases_mask = {
                'cov1': np.ones([20]),
                'cov2': np.ones([50]),
                'fc1': np.ones([500]),
                'fc2': np.ones([10])
            }

        mnist = input_data.read_data_sets("MNIST.data/", one_hot = True)
        # tf Graph input
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])
        lr = tf.placeholder(tf.float32, shape = [])

        keep_prob = tf.placeholder(tf.float32)
        keys = ['cov1','cov2','fc1','fc2']

        x_image = tf.reshape(x,[-1,28,28,1])
        # model number is iter_cnt - 1
        if (SAVE):
            init_file_part = model_number
        else:
            init_file_part = file_name_part

        (weights, biases) = initialize_variables(parent_dir, init_file_part)
        weights_new = {}
        for key in keys:
            weights_new[key] = weights[key] * tf.constant(weights_mask[key], dtype=tf.float32)

        # Construct model
        pred, pool = conv_network(x_image, weights_new, biases, keep_prob)

        # Define loss and optimizer
        trainer = tf.train.AdamOptimizer(learning_rate=lr)
    	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

        correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # I need to fetch this value
        variables = [weights['cov1'], weights['cov2'], weights['fc1'], weights['fc2'],
                    biases['cov1'], biases['cov2'], biases['fc1'], biases['fc2']]
        org_grads = trainer.compute_gradients(cost, var_list = variables, gate_gradients = trainer.GATE_OP)

        org_grads = [(ClipIfNotNone(grad), var) for grad, var in org_grads]
        # new_grads = mask_gradients(weights, org_grads, weights_mask, biases)

        # gradients are not masked
        train_step = trainer.apply_gradients(org_grads)


        init = tf.initialize_all_variables()
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            keys = ['cov1','cov2','fc1','fc2']

            _ = prune_info(weights, biases, 1)
            training_cnt = 0
            pruning_cnt = 0
            train_accuracy = 0
            accuracy_list = np.zeros(30)
            accuracy_mean = 0
            c = 0
            train_accuracy = 0
            batch_size = 128
            training_epochs = 12
            if (TRAIN == True):
                print('Training starts ...')
                for epoch in range(training_epochs):
                    avg_cost = 0.
                    total_batch = int(mnist.train.num_examples/batch_size)
                    # Loop over all batches
                    for i in range(total_batch):
                        # execute a pruning
                        batch_x, batch_y = mnist.train.next_batch(batch_size)
                        _ = sess.run(train_step, feed_dict = {
                                x: batch_x,
                                y: batch_y,
                                lr: learning_rate,
                                keep_prob: dropout})
                        training_cnt = training_cnt + 1
                        if (training_cnt % 10 == 0):
                            [c, train_accuracy] = sess.run([cost, accuracy], feed_dict = {
                                x: batch_x,
                                y: batch_y,
                                lr: learning_rate,
                                keep_prob: 1.})
                            accuracy_list = np.concatenate((np.array([train_accuracy]),accuracy_list[0:29]))
                            accuracy_mean = np.mean(accuracy_list)
                            if (training_cnt % 1000 == 0):
                                print('accuracy mean is {}'.format(accuracy_mean))
                                print('Epoch is {}'.format(epoch))
                                perc = prune_info(weights_new, biases, 0)
                        if (accuracy_mean > 0.99 or epoch > 10):
                            accuracy_list = np.zeros(30)
                            accuracy_mean = 0
                            print('Training ends')
                            test_accuracy = accuracy.eval({
                                    x: mnist.test.images[:],
                                    y: mnist.test.labels[:],
                                    lr: learning_rate,
                                    keep_prob: 1.})
                            perc = prune_info(weights_new, biases, 0)
                            print('test accuracy is {}'.format(test_accuracy))
                            print('crates are: {}'.format(crate))
                            if (epoch % 300 == 0):
                                learning_rate = learning_rate / float(10)
                            if (test_accuracy > 0.9936 or epoch > 10):
                                file_name_part = compute_file_name(crate)
                                file_name = parent_dir + 'weight_crate' + file_name_part + '.pkl'
                                with open(file_name, 'wb') as f:
                                    pickle.dump((
                                        weights['cov1'].eval(),
                                        weights['cov2'].eval(),
                                        weights['fc1'].eval(),
                                        weights['fc2'].eval(),
                                        biases['cov1'].eval(),
                                        biases['cov2'].eval(),
                                        biases['fc1'].eval(),
                                        biases['fc2'].eval()),f)
                                mask_info(weights_mask)
                                return (test_accuracy, perc)
                            else:
                                pass
                        # Compute average loss
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
                print("Optimization Finished!")
                batch_size = 128
                total_batch = int(mnist.test.num_examples/batch_size)
                acc_list = []
                for i in range(total_batch):
                    batch_x, batch_y = mnist.test.next_batch(batch_size)
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    test_accuracy = accuracy.eval({x: batch_x, y: batch_y, keep_prob : 1.0})
                    acc_list.append(test_accuracy)
                print("Accuracy:", np.mean(acc_list))
                with open('acc_log_10.txt','a') as f:
                    f.write(str(test_accuracy)+'\n')
                prune_perc = prune_info(weights_new, biases, 0)
                return (np.mean(acc_list), prune_perc)
                # Test model
                correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            if (TRAIN == False):
                if (PRUNE_ONLY == True):
                    mask_info(weights_mask)
                    prune_weights(weights, biases, weights_mask, crate, iter_cnt, parent_dir)
                if (SAVE == True):
                    file_name_part = compute_file_name(crate)
                    mask_file_name = parent_dir+'mask_crate'+ file_name_part+'.pkl'
                    file_name = parent_dir+'weight_crate'+ file_name_part+'.pkl'
                    print("saving for next iteration's computation"  + mask_file_name)
                    with open(mask_file_name, 'wb') as f:
                        pickle.dump(weights_mask, f)
                    mask_info(weights_mask)
                    with open(file_name, 'wb') as f:
                        pickle.dump((
                            weights['cov1'].eval(),
                            weights['cov2'].eval(),
                            weights['fc1'].eval(),
                            weights['fc2'].eval(),
                            biases['cov1'].eval(),
                            biases['cov2'].eval(),
                            biases['fc1'].eval(),
                            biases['fc2'].eval()),f)

                # Calculate accuracy
                batch_size = 128
                total_batch = int(mnist.test.num_examples/batch_size)
                acc_list = []
                print(total_batch)
                for i in range(total_batch):
                    batch_x, batch_y = mnist.test.next_batch(batch_size)
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    test_accuracy = accuracy.eval({x: batch_x, y: batch_y, keep_prob : 1.0})
                    acc_list.append(test_accuracy)
                print("Accuracy:", np.mean(acc_list))
                with open('acc_log_10.txt','a') as f:
                    f.write(str(test_accuracy)+'\n')
                prune_perc = prune_info(weights_new, biases, 0)
                return (np.mean(acc_list), prune_perc)

    except Usage, err:
        print >> sys.stderr, err.msg
        print >> sys.stderr, "for help use --help"
        return 2

def weights_info(iter,  c, train_accuracy, acc_mean):
    print('This is the {}th iteration, cost is {}, accuracy is {}, accuracy mean is {}'.format(
        iter,
        c,
        train_accuracy,
        acc_mean
    ))

def prune_info(weights, biases, counting):
    t_non_zeros = 0
    t_total = 0
    if (counting == 0):
        (non_zeros, total) = calculate_non_zero_weights(weights['cov1'].eval())
        (non_zeros_b, total_b) = calculate_non_zero_weights(biases['cov1'].eval())
        t_total += total + total_b
        t_non_zeros += non_zeros + non_zeros_b
        print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
        (non_zeros, total) = calculate_non_zero_weights(weights['cov2'].eval())
        (non_zeros_b, total_b) = calculate_non_zero_weights(biases['cov2'].eval())
        t_total += total + total_b
        t_non_zeros += non_zeros + non_zeros_b
        print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/float(total)))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'].eval())
        (non_zeros_b, total_b) = calculate_non_zero_weights(biases['fc1'].eval())
        t_total += total + total_b
        t_non_zeros += non_zeros + non_zeros_b
        print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/float(total)))
        (non_zeros, total) = calculate_non_zero_weights(weights['fc2'].eval())
        (non_zeros_b, total_b) = calculate_non_zero_weights(biases['fc1'].eval())
        t_total += total + total_b
        t_non_zeros += non_zeros + non_zeros_b
        print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/total))
    if (counting == 1):
        (non_zeros, total) = calculate_non_zero_weights(weights['fc1'].eval())
        print('take fc1 as example, {} nonzeros, in total {} weights'.format(non_zeros, total))
    if (counting == 1):
        perc = 0
    else:
        perc = t_non_zeros / float(t_total)
    print("non zeros {}, total {}".format(t_non_zeros, t_total))
    print("perc {}".format(perc))
    return perc

def mask_info(weights):
    (non_zeros, total) = calculate_non_zero_weights(weights['cov1'])
    print('cov1 has prunned {} percent of its weights'.format((total-non_zeros)*100/float(total)))
    (non_zeros, total) = calculate_non_zero_weights(weights['cov2'])
    print('cov2 has prunned {} percent of its weights'.format((total-non_zeros)*100/float(total)))
    (non_zeros, total) = calculate_non_zero_weights(weights['fc1'])
    print('fc1 has prunned {} percent of its weights'.format((total-non_zeros)*100/float(total)))
    (non_zeros, total) = calculate_non_zero_weights(weights['fc2'])
    print('fc2 has prunned {} percent of its weights'.format((total-non_zeros)*100/float(total)))

def write_numpy_to_file(data, file_name):
    # Write the array to disk
    with file(file_name, 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(data.shape))

        for data_slice in data:
            for data_slice_two in data_slice:
                np.savetxt(outfile, data_slice_two)
                outfile.write('# New slice\n')


if __name__ == '__main__':
    sys.exit(main())
