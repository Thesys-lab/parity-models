import pickle
import random


def getdata():
    # read data and shuffle
    index=[i for i in range(11788)]
    random.shuffle(index)

    f=open("./id_to_data","rb+")
    data=pickle.load(f)
    data=data[index]
    data_train=data[0:9000]
    data_test=data[9000:]
    f=open("./id_to_box","rb+")
    box=pickle.load(f)
    box=box[index]
    box_train=box[0:9000]
    box_test=box[9000:]
    return data_train,box_train,data_test,box_test
