from keras import Model
from keras import backend
from keras.layers import Input, Dense, Concatenate
from keras.losses import binary_crossentropy

from data_generator import MnistDoubleGenerator


EPOCHS = 5
WORKERS = 4


def train_verifier():

    #first define our network

    two_encodings = Input([512])
    # a = Dense(64, activation="sigmoid")(two_encodings)
    # b = Dense(64, activation="sigmoid")(a)
    output = Dense(1,activation="sigmoid")(two_encodings)

    verifier = Model(inputs=[two_encodings], outputs=[output])
    verifier.compile(optimizer="adam",loss="mean_squared_error",metrics=['binary_accuracy'])
    verifier.summary()

    # get our data generator
    train_generator = MnistDoubleGenerator(4096,"./mnist/fashion_precomp_encoderBII_train.csv")

    # for x, y  in train_generator:
    #     print(x)
    #     print(y)

    verifier.fit_generator(
        train_generator,
        epochs=EPOCHS,
    )

    test_generator = MnistDoubleGenerator(4096, "mnist/fashion_precomp_encoderBII_test.csv")

    eval_acc = verifier.evaluate_generator(
        test_generator
    )

    print(eval_acc)


if __name__ == "__main__":
    train_verifier()