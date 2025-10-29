
######### 划分数据集 split dataset into train/test

def split_data(data, proportion):
    x = data[:,:7]
    y = data[:,7:]

    x_train = x[:int(data.shape[0]*proportion), :]
    x_test = x[int(data.shape[0]*proportion):, :]

    y_train = y[:int(data.shape[0]*proportion), :]
    y_test = y[int(data.shape[0]*proportion):, :]

    return x_train, y_train, x_test, y_test

