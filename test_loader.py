import os

def test_loader(path):
    i = 0
    data_dic = []
    dir_busy = path + '/busy/'
    dir_free = path + '/free/'

    imglist = os.listdir(dir_busy) + os.listdir(dir_free)
    for item in imglist:
        if i < len(os.listdir(dir_busy)):
            img_dic = {'img_path': dir_busy + item, 'label': 1}
        else:
            img_dic = {'img_path': dir_free + item, 'label': 0}
        data_dic.append(img_dic)
        i += 1
        if i > 5:
            break
    return data_dic
list1 = [1, 2]

test_data = test_loader('/home/stefan/parkingLot/validation/')
for item in test_data:
    print(test_data.index(item))
