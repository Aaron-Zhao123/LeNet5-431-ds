import os
import training_ds
# os.system('python training_v3.py -p0')
# os.system('python training_v3.py -p1')
# os.system('python training_v3.py -p2')
# os.system('python training_v3.py -p3')
# os.system('python training_v3.py -p4')
# os.system('python training_v3.py -p4')
# os.system('python training_v3.py -p5')

acc_list = []
count = 0
pcov = 90
pfc = 99.5
pcov2 = 96
pfc2 = 96
pcov = 0
pfc = 99.9
pcov2 = 0
pfc2 = 0
# model_tag = 'pcov'+str(pcov)+'pcov'+str(pcov2)+'pfc'+str(pfc)+'pfc'+str(pfc2)
# pfc = pfc+1
# param = [
# ('-pcov',pcov),
# ('-pcov2',pcov2),
# ('-pfc',pfc),
# ('-pfc2',pfc2),
# ('-m',model_tag),
# ('-ponly', True),
# ('-test', False)
# ]
# acc = training_v6.main(param)
model_tag = 'pcov'+str(pcov)+'pcov'+str(pcov2)+'pfc'+str(int(round(pfc*10)))+'pfc'+str(pfc2)
retrain_cnt = 0
learning_rate = 1e-4
while (count < 10):
    if (retrain_cnt == 0):
        pcov2 = pcov2 + 10
        # pfc = pfc + 0.1
    # pruning
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
    ('-recover_rate', 0.005)
    ]
    _ = training_ds.main(param)

    model_tag = 'pcov'+str(pcov)+'pcov'+str(pcov2)+'pfc'+str(int(round(pfc*10)))+'pfc'+str(pfc2)
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
    ('-recover_rate', 0.005)
    ]
    acc = training_ds.main(param)
    acc_list.append(acc)
    if (acc < 0.9936):
        retrain_cnt += 1
        if (retrain_cnt % 2 == 0):
            learning_rate = learning_rate / 2
        if (retrain_cnt > 5):
            learning_rate = 1e-4
            pass
    else:
        retrain_cnt = 0
        count = count + 1
#
# param = [
# ('-pcov',pcov),
# ('-pcov2',pcov2),
# ('-pfc',pfc),
# ('-pfc2',pfc2),
# ('-m',model_tag)
# ]
# acc = training_ds.main(param)
#
# while (count < 10):
#     pfc = pfc + 1
#     pfc2 = pfc2 + 1
#     pcov = pcov + 1
#     pcov = pcov2 + 1
#     param = [
#     ('-pcov',pcov),
#     ('-pcov2',pcov2),
#     ('-pfc',pfc),
#     ('-pfc2',pfc2),
#     ('-m',model_tag)
#     ]
#     acc = training_v6.main(param)
#     model_tag = 'pcov'+str(pcov)+'pcov'+str(pcov2)+'pfc'+str(pfc)+'pfc'+str(pfc2)
#     acc_list.append(acc)
#     count = count + 1
#     if (acc < 0.99):
#         break
# print (acc)
#
# print('accuracy summary: {}'.format(acc_list))
