import pandas as pd
import numpy as np
import code

from keras.models import load_model, Model
from keras.layers import Input, Reshape
from data_generator import MnistSingleGenerator
from sanity_check import sanity_check_encoding

KEY_STR = "MNIST-E1.h5"
TYPE_STR = "train"


def compute_embedding(model_path,batch_size, data_path):
    # load our model
    encoder = load_model(model_path)
    input_tensor = Input((784,))
    reshaped = Reshape((28,28,1))(input_tensor)
    embedding = encoder(reshaped)

    encoder_ready = Model(inputs=[input_tensor],output=[embedding])

    data_gen = MnistSingleGenerator(batch_size, data_path)

    output = pd.DataFrame()
    for batch_x, batch_y in data_gen:
        encoding = pd.DataFrame(encoder_ready.predict_on_batch(batch_x))

        computed_encoding = pd.concat([batch_y,encoding],axis=1,join="outer",sort=False)
        output = output.append(computed_encoding,ignore_index=True)

    return output


if __name__ == "__main__":
    output = compute_embedding(f"trained_models/{KEY_STR}",512,f"./mnist/{TYPE_STR}.csv")
    output.to_csv(f"mnist/{KEY_STR}_{TYPE_STR}.csv", index=False)
    sanity_check_encoding(output,key_str=KEY_STR)





