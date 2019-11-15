import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
import uuid

from keras.models import Model, load_model
from keras import backend
from keras.layers import Input, Dense, Concatenate, Reshape, MaxPool2D
from keras.callbacks import EarlyStopping


from data_generator import MnistDoubleGenerator, MnistDemoGenerator


EPOCHS = 50
WORKERS = 6


def train_verifier():

    #first define our network
    two_encodings = Input((512,))
    a = Dense(64, activation="selu")(two_encodings)
    b = Dense(64, activation="selu")(a)
    c = Dense(32, activation="relu")(b)
    d = Dense(32, activation="relu")(c)
    e = Dense(32, activation="relu")(d)
    output = Dense(1,activation="sigmoid")(e)

    verifier = Model(inputs=[two_encodings], outputs=[output])
    verifier.compile(optimizer="Nadam",loss="binary_crossentropy",metrics=['binary_accuracy'])
    verifier.summary()

    # get our data generator
    train_generator = MnistDoubleGenerator(4096,"mnist/MNIST-E1-t1573849381-a10.h5_train.csv")

    # for x, y  in train_generator:
    #     print(x)
    #     print(y)

    verifier.fit_generator(
        train_generator,
        epochs=EPOCHS,
        # callbacks=[EarlyStopping("binary_accuracy")],
        use_multiprocessing=True,
        workers=WORKERS
    )

    test_generator = MnistDoubleGenerator(4096, "mnist/MNIST-E1-t1573849381-a10.h5_train.csv")

    eval_acc = verifier.evaluate_generator(
        test_generator
    )

    verifier.save(f"./saved_models/MNIST-V1-{int(time.time())}-{eval_acc[1]}.h5")


def visually_evaluate(encoder_path, verifier_path, batch_count, lucky_number=0):
    # first define our model
    encoder = load_model(encoder_path)
    verifier = load_model(verifier_path)

    pic_a = Input((28,28,1))
    pic_b = Input((28,28,1))

    encoding_a = encoder(pic_a)
    encoding_b = encoder(pic_b)

    two_encodings = Concatenate(axis=1)([encoding_a,encoding_b])
    result = verifier(two_encodings)

    runtime_model = Model(inputs=[pic_a, pic_b], outputs=[result])
    runtime_model.compile(optimizer="adam",loss="mean_squared_error",metrics=['accuracy'])

    data_generator = MnistDemoGenerator(32,"./mnist/train.csv",batch_count=batch_count)

    for x_1, x_2, y in data_generator:
        image_1 = np.reshape(x_1,(-1,28,28,1))
        image_2 = np.reshape(x_2,[-1,28,28,1])

        bianry_result = runtime_model.predict_on_batch([image_1,image_2])

        truth = y[lucky_number]
        raw = bianry_result[lucky_number]
        rounded = np.round(bianry_result)[lucky_number]
        output_string = f"result : raw={raw},rounded={rounded},truth={truth}"
        print(output_string)

        if int(rounded) != int(truth) or rounded == 1.0:
        # if True:
            plt.figure(1)
            plt.title(output_string)
            plt.subplot(211)
            plt.imshow(image_1[lucky_number,:,:,0])
            plt.subplot(212)
            plt.imshow(image_2[lucky_number,:,:,0])
            plt.savefig(f"./visualizations/better_digits-{uuid.uuid4()}-{int(rounded) == int(truth) and 'right' or 'wrong' }.png")
    return




if __name__ == "__main__":
    # train_verifier()
    visually_evaluate(
        "trained_models/MNIST-E1-t1573849381-a10.h5",
        "saved_models/MNIST-V1-1573849639-0.893505871295929.h5",
        100
    )
    pass