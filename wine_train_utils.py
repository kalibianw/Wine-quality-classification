from tensorflow.keras import models, layers, activations, optimizers, losses, callbacks
import matplotlib.pyplot as plt
import shutil
import os


def create_model(input_dim):
    model = models.Sequential([
        layers.Dense(4096, activation=activations.relu, input_dim=input_dim),
        layers.Dropout(rate=0.5),
        layers.Dense(2048, activation=activations.relu),
        layers.Dense(2048, activation=activations.relu),
        layers.Dropout(rate=0.5),
        layers.Dense(1024, activation=activations.relu),
        layers.Dense(1024, activation=activations.relu),
        layers.Dropout(rate=0.5),
        layers.Dense(512, activation=activations.relu),
        layers.Dense(128, activation=activations.relu),

        layers.Dense(5, activation=activations.softmax),
    ])

    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.categorical_crossentropy,
        metrics=['acc']
    )

    return model


def training(model, x_train, y_train, x_valid, y_valid, ckpt_path):
    shutil.rmtree('training_log')
    os.mkdir('training_log')

    monitor = 'val_acc'

    history = model.fit(
        x=x_train, y=y_train,
        batch_size=32,
        epochs=500,
        callbacks=[
            callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                monitor=monitor,
                verbose=2,
                save_best_only=True,
                save_weights_only=True
            ),
            callbacks.EarlyStopping(
                monitor=monitor,
                min_delta=1e-4,
                patience=50,
                verbose=2
            ),
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                verbose=1,
                min_lr=1e-3
            ),
            callbacks.TensorBoard(
                log_dir='training_log',
                write_images=True
            )
        ],
        validation_data=(x_valid, y_valid),
        shuffle=True
    )

    return history


def training_visualization(hist):
    plt.subplot(2, 1, 1)
    plt.plot(hist['acc'])
    plt.plot(hist['val_acc'])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracies")

    plt.subplot(2, 1, 2)
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
