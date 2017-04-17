import os
import training_ds_crate

acc_list = []
count = 0
pcov = 0
pfc = 0
pcov2 = 0
pfc2 = 0
retrain_cnt = 0
learning_rate = 1e-5
recover_rates = [0,0.1,0.002,0]
model_tag = 0
count = 0
crates = {
    'cov1':1,
    'cov2':1,
    'fc1':3,
    'fc2':1
}
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
    ('-parent_dir', './'),
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
    ('-parent_dir', './'),
    ('-recover_rate', recover_rates),
    ('-iter_cnt', count),
    ('-crate',crates)
    ]
    acc = training_ds_crate.main(param)
    acc_list.append(acc)
    print('acc summary is {}'.format(acc_list))
