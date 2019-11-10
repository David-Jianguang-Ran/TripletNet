from keras import Model
from keras.layers import Input, Dense, Concatenate

from data_generator import MnistDoubleGenerator


EPOCHS = 10
WORKERS = 4

def train_verifier():

    #first define our network

    two_encodings = Input([512])
    a = Dense(256,activation="elu",name="fully_connected_1")(two_encodings)
    b = Dense(128,activation="elu",name="fully_connected_2")(a)
    output = Dense(1,activation="elu")(b)

    verifier = Model(inputs=[two_encodings], outputs=[output])
    verifier.compile(optimizer="adagrad",loss="squared_hinge",metrics=['accuracy'])
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