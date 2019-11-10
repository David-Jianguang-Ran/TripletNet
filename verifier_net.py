from keras import Model
from keras.layers import Input, Dense, Concatenate

from data_generator import MnistDoubleGenerator


EPOCHS = 100
WORKERS = 4

def train_verifier():

    #first define our network

    two_encodings = Input([512])
    b = Dense(64, activation="relu", name="fully_connected_2")(two_encodings)
    output = Dense(1,activation="sigmoid")(b)

    verifier = Model(inputs=[two_encodings], outputs=[output])
    verifier.compile(optimizer="SGD",loss="hinge",metrics=['accuracy'])
    verifier.summary()

    # get our data generator
    train_generator = MnistDoubleGenerator(2048,"./mnist/fashion_precomp_encoderIII_train.csv")


    verifier.fit_generator(
        train_generator,
        epochs=EPOCHS,
    )

    test_generator = MnistDoubleGenerator(2048, "mnist/fashion_precomp_encoderIII_test.csv")

    eval_acc = verifier.evaluate_generator(
        test_generator
    )

    print(eval_acc)


if __name__ == "__main__":
    train_verifier()