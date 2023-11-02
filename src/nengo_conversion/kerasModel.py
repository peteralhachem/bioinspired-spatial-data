import tensorflow.keras as tk
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

def plot_training(train_accuracy, val_accuracy, train_loss, val_loss):

    epochs = range(1, len(train_accuracy) + 1)

    # Plot dell'accuratezza
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracy, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r*-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot della loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('loss_accuracy_keras.jpg')
    
def train_keras(model, train, validation, test, batch_size):

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.02944231878048946,
                decay_steps=10000,
                decay_rate=float(0.0005554738683782042))
    
    
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate = lr_schedule, momentum = float(0.8886002470522787), nesterov = True),
                    loss='categorical_crossentropy',                          
                    metrics=['accuracy'])
    
    checkpoint_callback = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    history = model.fit(x = train[0], 
                y = train[1],
                batch_size = batch_size,
                epochs = 49,
                steps_per_epoch=(train[0].shape[0]//batch_size),
                validation_data=(validation[0], validation[1]),
                callbacks=[checkpoint_callback]
                )
    
    results = model.evaluate(x=test[0], y=test[1], batch_size = batch_size)
    print("test loss, test acc", results)

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    plot = False

    if plot:
        plot_training(train_accuracy, val_accuracy, train_loss, val_loss)
    

def kmodel(train, validation, test):


    layers_gen5_id1 = [
        tk.layers.AveragePooling1D(pool_size=61, strides=3, padding='same'),
         tk.layers.Conv1D(filters=131, kernel_size=29,
                    strides=3,padding='valid' ,activation='relu', use_bias = False,
                    kernel_initializer='he_normal'),
         tk.layers.Conv1D(filters=20, kernel_size=44,
                    strides=5,padding='same' ,activation='relu', use_bias = True,
                    kernel_initializer='he_normal'),
         tk.layers.Flatten(),
         tk.layers.Dense(168, activation='linear', use_bias=True, kernel_initializer='he_normal'),
         tk.layers.Dense(10, activation='softmax', use_bias=True, kernel_initializer='he_normal')
        
    ]


    data_layer = []

    inputs = tk.layers.Input(shape = (1881,1))
    x = inputs

    for l in layers_gen5_id1:
        x = l(x)
        data_layer.append(x)

    model = tk.models.Model(inputs=inputs, outputs=data_layer[-1])

    try:
        urlretrieve(
        "https://drive.google.com/uc?export=download&"
        "id=1uv8U1gZ2IyP1oJdkL_0e0hrKZF6oRK1P",	
        "best_model.h5",
        )
    except:
        train_keras(model, train, validation, test, batch_size = 137)

    model.load_weights('best_model.h5')

    return inputs, data_layer[1], data_layer[3], data_layer[-1], model