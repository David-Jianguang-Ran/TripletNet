import time

from keras import backend
from keras.models import Model
from keras.layers import Conv2D, Concatenate, MaxPool2D, Flatten, Input, Reshape, Lambda, AveragePooling2D, Activation
from keras.callbacks import EarlyStopping

from data_generator import MnistTripletGenerator


# Settings
DATA_PATH = "./mnist/train.csv"

BATCH_SIZE = 256
PRE_PROCESS_WORKERS = 6


def get_triplet_loss_with_arg(embedding_length=256, alpha=1.0):
  
  def distance_function(v1, v2):
    '''
    :param v1:
    :param v2:
    :return: real number distance
    '''
    return backend.sum(backend.square(v1 - v2), axis=1)

  def triplet_loss(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:  shape = (? x embedding*3)
    :param alpha:
    :return:
    """
    embedding_anchor = backend.slice(y_pred,[0,0],[-1,embedding_length])
    embedding_positive = backend.slice(y_pred,[0,embedding_length],[-1,embedding_length])
    embedding_negative = backend.slice(y_pred,[0,embedding_length*2],[-1,embedding_length])
    
    distance_positive = distance_function(embedding_anchor,embedding_positive)
    distance_negative = distance_function(embedding_anchor,embedding_negative)
    
    loss = distance_positive - distance_negative + alpha
    return backend.maximum(loss, 0.0)

  return triplet_loss


def test_train_network(alpha=1.0,epochs=2):
  
  def add_inception_module(input_node, layer_name_prefix):
    """
    :param input_node:
    :param input_channels: tuple (-1 , height, width, channels)
    :param layer_name_prefix:
    :return:
    """
    
    one_by_one = Conv2D(64, (1, 1), activation="relu", name=layer_name_prefix + "_1x1")(input_node)
    
    a = Conv2D(96, (1, 1), activation='relu', name=layer_name_prefix + "_pre_3x3")(input_node)
    three_by_three = Conv2D(128,(3,3),padding="same",activation="relu",name=layer_name_prefix+"_3x3")(a)
    
    b = Conv2D(16, (1, 1), activation='relu', name=layer_name_prefix + "_pre_5x5")(input_node)
    five_by_five = Conv2D(32,(5,5),padding="same",activation="relu",name=layer_name_prefix+"_5x5")(b)
    
    c = MaxPool2D((3,3),padding="same",strides=1)(input_node)
    pooled = Conv2D(32, (1, 1), activation="relu", name=layer_name_prefix + "_pooled")(c)

    concat_output = Concatenate(axis=3)([one_by_one,three_by_three,five_by_five,pooled])
    return concat_output
  
  # lets make our encoder network
  shared_model_input = Input((28,28,1,))
  a = add_inception_module(shared_model_input,layer_name_prefix="inception_first")
  b = MaxPool2D((4,4),padding="same")(a)
  c = add_inception_module(b,layer_name_prefix="inception_second")
  d = MaxPool2D((3,3),padding="same")(c)
  e = add_inception_module(d,layer_name_prefix="inception_third")
  f = Activation("sigmoid")(MaxPool2D((3,3),padding="same")(e))
  g = Flatten()(f)
  
  encoder_network = Model(inputs=shared_model_input,outputs=g,name="shared_encoder_network")
  encoder_network.summary()

  # split the input into three seperate images
  def _get_slicer_at(index):
    '''

    :param index: int encoding fist second or third slice of the original input
    :return: function for slicing
    '''
    def _slicer(input_tensor):
      return backend.slice(input_tensor,(0, 784 * index), (-1,784))
    return _slicer

  concat_input = Input((2352,))
  anchor = Reshape((28,28,1))(Lambda(_get_slicer_at(0),name="slicer_a")(concat_input))
  positive = Reshape((28,28,1))(Lambda(_get_slicer_at(1),name='slicer_p')(concat_input))
  negative = Reshape((28,28,1))(Lambda(_get_slicer_at(2),name='slicer_n')(concat_input))

  # feed each image through the encoder
  anchor_encoded = encoder_network(anchor)
  positive_encoded = encoder_network(positive)
  negative_encoded = encoder_network(negative)

  # output concatenated encodings
  predictions = Concatenate(axis=1)([anchor_encoded,positive_encoded,negative_encoded])
  
  model = Model(inputs=[concat_input],outputs=[predictions])
  model.compile(loss=get_triplet_loss_with_arg(alpha=alpha), optimizer='adam')
  model.summary()

  test_data_generator = MnistTripletGenerator(BATCH_SIZE,DATA_PATH)

  loss = model.fit_generator(
    generator=test_data_generator,
    workers=PRE_PROCESS_WORKERS,
    use_multiprocessing=True,
    epochs=epochs,
    callbacks=[EarlyStopping(monitor="loss")]
  )

  test_eval_data_generator = MnistTripletGenerator(BATCH_SIZE,"./mnist/fashion-mnist_test.csv")

  eval_loss = model.evaluate_generator(
    generator=test_eval_data_generator,
    workers=PRE_PROCESS_WORKERS,
    use_multiprocessing=True
  )
  save_path = "./trained_models/MNIST-E1-t{}-a{}-l{].h5".format(int(time.time()),alpha,eval_loss[1])
  encoder_network.save(save_path)

  return save_path


