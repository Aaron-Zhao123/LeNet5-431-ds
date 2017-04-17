import os
import training_ds_crate
from shutil import copyfile

acc_list = []
count = 0
pcov = 0
pfc = 0
pcov2 = 0
pfc2 = 0
retrain_cnt = 0
learning_rate = 1e-5
model_tag = 0
count = 0
crates = {
    'cov1': 0,
    'cov2': 0,
    'fc1':1,
    'fc2':0
}
while (desired_crates < 3):
    parent_dir = './assets/' + 'crfc1v' + str(int(crates['fc1']*10)) + '/'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        src_dir = prev_parent_dir+'weight_crate'+str(count)+'.pkl'
        dest_dir = parent_dir + 'weight_crate0.pkl'
        copyfile(src_dir,dest_dir)
    count = 0
    model_tag = 0
    while (count <= 5):
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
        ('-recover_rate', recover_rates),
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
        ('-recover_rate', recover_rates),
        ('-iter_cnt', count),
        ('-crate',crates)
        ]
        acc = training_ds_crate.main(param)
        acc_list.append(acc)
        if (acc > 0.9946):
            break
        print('acc summary is {}'.format(acc_list))
    prev_parent_dir = './assets/' + 'crfc1v' + str(crates['fc1']) + '/'
    crates['fc1'] += 0.2
