import numpy as np
import csv
import os


def open_wine_csv(train_path):
    if os.path.exists(train_path) is False:
        print("Checking train_path")
        exit()
    raw_train = list()
    frow = list()
    is_first = True

    with open(train_path) as handler:
        reader = csv.reader(handler)
        for row in reader:
            if is_first is True:
                frow.append(row)
                is_first = False
            else:
                raw_train.append(row)

    raw_train = np.array(raw_train)
    label = raw_train[:, 1]
    label = label.astype('int')
    index = raw_train[:, 0]
    index = index.astype('int')
    frow = np.array(frow)

    return raw_train, frow, label, index


def preprocessing_2(raw_data, is_train):
    row = [0, 3, 5, 6, 7, 8, 9, 10]
    wine_type = raw_data[:, -1]

    for i in range(len(wine_type)):
        if str(wine_type[i]) == 'red':
            wine_type[i] = 0
        elif str(wine_type[i]) == 'white':
            wine_type[i] = 1
        else:
            wine_type[i] = 2

    if 2 in wine_type:
        print('화이트도, 레드도 아닌 와인이 존재합니다.\n'
              '학습 진행에는 문제가 없지만 학습 과정에 문제가 발생할 수 있습니다.\n'
              '데이터 셋을 확인 해 주세요'
              )

    if is_train is True:
        raw_data = raw_data[:, 2:-1]
    wine_type = wine_type.reshape((len(wine_type), 1))

    for i in range(len(row)):
        tmp = raw_data[:, row[i]]
        tmp = tmp.astype('float32')
        max_value = np.max(tmp)
        tmp = tmp / max_value
        raw_data[:, row[i]] = tmp

    raw_data = np.append(raw_data, wine_type, axis=1)
    raw_data = raw_data.astype('float32')

    return raw_data
