from tensorflow.keras import models
import wine_data_utils as utils
import numpy as np
import csv

model_save_path = "wine.h5"

model = models.load_model(model_save_path)
model.summary()

with open(r'Data\test.csv', 'r') as handler:
    reader = csv.reader(handler)
    raw_test = list()
    next(reader)
    for row in reader:
        raw_test.append(row)

    raw_test = np.array(raw_test)
    raw_test = raw_test[:, 1:]

    utils.preprocessing_2(raw_data=raw_test, is_train=False)

    print(raw_test[0, :])

raw_test = raw_test.astype('float32')
pred = model.predict(raw_test)
print(np.shape(pred))

pred_index = list()
for i in range(len(pred[:, 0])):
    pred_index.append(np.argmax(pred[i, :]))

print(np.shape(pred_index))
print(pred_index)
