import os
import sys
import training_ds_crate
from shutil import copyfile

def compute_file_name(p):
    name = ''
    name += 'cov' + str(int(p['cov1'] * 10))
    name += 'cov' + str(int(p['cov2'] * 10))
    name += 'fc' + str(int(p['fc1'] * 10))
    name += 'fc' + str(int(p['fc2'] * 10))
    return name

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
crates = {
    'cov1': 0.,
    'cov2': 0.,
    'fc1': 0.,
    'fc2': 0.
}
parent_dir = 'assets_no_retrain/'

while (crates['fc1'] < 3.):
    count = 0
    model_tag = 0

    # pruning
    count = count + 1
    param = [
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
    # get test acc
    param = [
    ('-m',model_tag),
    ('-learning_rate',learning_rate),
    ('-prune', False),
    ('-train', False),
    ('-parent_dir', parent_dir),
    ('-iter_cnt', count),
    ('-crate',crates)
    ]
    acc = training_ds_crate.main(param)
    if (acc > 0.9936):
        print('acc passed...')
        break
    print('acc summary is {}'.format(acc_list))

    # save the model
    model_tag = compute_file_name(crates)
    crates['fc1'] += 1.
    # crates['cov1'] += .2
    # crates['fc2'] += .2
    param = [
    ('-m',model_tag),
    ('-learning_rate',learning_rate),
    ('-prune', False),
    ('-train', False),
    ('-next_iter_save', True),
    ('-parent_dir', parent_dir),
    ('-iter_cnt', count),
    ('-crate',crates)
    ]
    _ = training_ds_crate.main(param)
    acc_list.append((acc,crates['fc1']))
