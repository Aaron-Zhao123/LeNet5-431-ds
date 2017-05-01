import os
import sys
import training_ds_crate
from shutil import copyfile

acc_list = []
count = 0
pcov = 0
pfc = 0
pcov2 = 0
pfc2 = 0
retrain_cnt = 0
learning_rate = 1e-4
model_tag = 0
count = 0
# crates = {
#     'cov1': 0.1,
#     'cov2': 1.8,
#     'fc1': 3.38,
#     'fc2': 0.1
# }

crates = {
    'cov1': 0.,
    'cov2': 0.,
    'fc1': 0.,
    'fc2': 0.
}

prev_parent_dir = './assets/' + 'cr' + 'cov1v' + str(int(crates['cov1']*10))+ 'cov2v' + str(int(crates['cov2']*10)) + 'fc1v' + str(int(crates['fc1']*100))  + 'fc2v' + str(int(crates['fc2']*10)) + '/'
crates['fc1'] += 1.
while (crates['fc1'] < 4.):
    # parent_dir = './assets/' + 'crfc1v' + str(int(crates['fc1']*100)) + '/'
    parent_dir = './assets/' + 'cr' + 'cov1v' + str(int(crates['cov1']*10))+ 'cov2v' + str(int(crates['cov2']*10)) + 'fc1v' + str(int(crates['fc1']*100))  + 'fc2v' + str(int(crates['fc2']*10)) + '/'

    # parent_dir = './assets/' + 'cr' + 'cov2v' + str(int(crates['cov2']*10)) + 'fc1v' + str(int(crates['fc1']*100)) + '/'
    if not os.path.exists(parent_dir):
        print('am i here')
        os.makedirs(parent_dir)
        src_dir = prev_parent_dir+'weight_crate'+str(count)+'.pkl'
        dest_dir = parent_dir + 'weight_crate0.pkl'
        copyfile(src_dir,dest_dir)
        print(src_dir)
        print(dest_dir)
        src_dir = prev_parent_dir+'mask_crate'+str(count)+'.pkl'
        dest_dir = parent_dir + 'mask_crate0.pkl'
        copyfile(src_dir,dest_dir)
    count = 0
    model_tag = 0
    while (count <= 7):
        # pruning
        count = count + 1
        param = [
        ('-pcov',pcov),
        ('-pcov2',pcov2),
        ('-pfc',pfc),
        ('-pfc2',pfc2),
        ('-m',model_tag),
        ('-learning_rate',learning_rate),
        ('-prune', True),
        ('-train', False),
        ('-parent_dir', parent_dir),
        ('-iter_cnt', count),
        ('-crate',crates)
        ]
        _ = training_ds_crate.main(param)

        # after pruning
        # train weights based on the pruned model
        model_tag = model_tag + 1
        # Train
        param = [
        ('-pcov',pcov),
        ('-pcov2',pcov2),
        ('-pfc',pfc),
        ('-pfc2',pfc2),
        ('-m',model_tag),
        ('-learning_rate',learning_rate),
        ('-prune', False),
        ('-train', True),
        ('-parent_dir', parent_dir),
        ('-iter_cnt', count),
        ('-crate',crates)
        ]
        acc = training_ds_crate.main(param)
        if (acc > 0.9936):
            print('acc passed...')
            break
        print('acc summary is {}'.format(acc_list))
    acc_list.append((acc,crates['fc1']))
    # prev_parent_dir = './assets/' + 'crfc1v' + str(int(crates['fc1']*100)) + '/'
    # prev_parent_dir = './assets/' + 'cr' + 'cov2v' + str(int(crates['cov2']*10)) + 'fc1v' + str(int(crates['fc1']*100)) + '/'
    prev_parent_dir = './assets/' + 'cr' + 'cov1v' + str(int(crates['cov1']*10))+ 'cov2v' + str(int(crates['cov2']*10)) + 'fc1v' + str(int(crates['fc1']*100))  + 'fc2v' + str(int(crates['fc2']*10)) + '/'
    crates['fc1'] += 1.
    # crates['cov1'] += 0.1
    # crates['fc2'] += 0.2
