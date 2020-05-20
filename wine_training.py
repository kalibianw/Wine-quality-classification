from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import wine_data_utils as utils
import wine_train_utils as wt
import numpy as np


train_path = r"Data\train.csv"
ckpt_path = r"ckpt\wine.ckpt"
model_save_path = "wine.h5"

train_size = 0.7

raw_train, frow, label, index = utils.open_wine_csv(train_path=train_path)

raw_train = utils.preprocessing_2(raw_data=raw_train, is_train=True)

label = label / 2
label = to_categorical(label)

x_train, x_valid, y_train, y_valid = train_test_split(raw_train, label, train_size=train_size)
x_train, x_valid, y_train, y_valid = np.array(x_train), np.array(x_valid), np.array(y_train), np.array(y_valid)

input_dim = len(x_train[0, :])
model = wt.create_model(input_dim=input_dim)
model.summary()

history = wt.training(
    model=model,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
    ckpt_path=ckpt_path
)

model.load_weights(ckpt_path)
model.save(model_save_path)

hist = history.history
wt.training_visualization(hist)
plt.show()
