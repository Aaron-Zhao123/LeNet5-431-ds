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
pfc = 99
pcov2 = 90
pfc2 = 90
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
model_tag = 'pcov'+str(pcov)+'pcov'+str(pcov2)+'pfc'+str(pfc)+str(count)+'pfc'+str(pfc2)
while (count < 10):
    pfc = pfc + 0.1
    # pfc2 = pfc2 + 10
    # pcov = pcov + 10
    # pcov2 = pcov2 + 10
    param = [
    ('-pcov',pcov),
    ('-pcov2',pcov2),
    ('-pfc',pfc),
    ('-pfc2',pfc2),
    ('-m',model_tag)
    ]
    acc = training_ds.main(param)
    model_tag = 'pcov'+str(pcov)+'pcov'+str(pcov2)+'pfc'+str(int(round(pfc*10)))+'pfc'+str(pfc2)
    acc_list.append(acc)
    count = count + 1
    if (acc < 0.99):
        break
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
