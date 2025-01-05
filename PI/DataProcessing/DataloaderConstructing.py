from torch.utils.data import Dataset,DataLoader,TensorDataset
import torch
import numpy as np
##This is for building dataset, it can be fed numpy arraies and give the dataloader##

def DataloaderConstructing(X_train,y_train,X_test,y_test,batch_size,
                      shuffle=True,pin_memory=True):
    y_train,y_test=StringToLabel(y_train),StringToLabel(y_test)
    x_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train.squeeze())
    x_test, y_test = torch.from_numpy(X_test), torch.from_numpy(y_test.squeeze())
    Deal_train_dataset, Deal_test_dataset = TensorDataset(x_train, y_train), \
                                            TensorDataset(x_test, y_test)
    Train_loader, Test_loader = DataLoader(dataset=Deal_train_dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           pin_memory=pin_memory,
                                           ), \
                                DataLoader(dataset=Deal_test_dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           pin_memory=pin_memory,
                                           )
    return Train_loader, Test_loader

def StringToLabel(y):
    labels=np.unique(y)
    new_label_list=[]
    for label in y:
        for position,StringLabel in enumerate(labels):
            if label==StringLabel:
                new_label_list.append(position)
            else:
                continue
    new_label_list=np.array(new_label_list)
    return new_label_list

