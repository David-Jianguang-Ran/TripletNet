import keras
import tensorflow
import code

from keras import backend
from keras.models import Model
from keras.layers import Conv2D, Concatenate, MaxPool2D, Flatten, Input, Reshape, Lambda

from triplet_maker_mnist import MnistTripletGenerator


def triplet_loss_with_arg(embedding_length, alpha=0.0):
  
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


def test_network_main():
  
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
  b = MaxPool2D((2,2),padding="same")(a)
  c = add_inception_module(b,layer_name_prefix="inception_second")
  d = MaxPool2D((2,2),padding="same")(c)
  e = add_inception_module(d,layer_name_prefix="inception_third")
  f = Flatten()(e)
  
  shared_model = Model(inputs=shared_model_input,outputs=f,name="shared_encoder_network")
  shared_model.summary()
  
  
  # # run each picture through the shared encoder network
  # anchor = Input((28,28,1,),name="anchor")
  # positive = Input((28,28,1,),name="positive")
  # negative = Input((28,28,1,),name="negative")

  concat_input = Input((2352,))

  def _get_slicer_at(order):
    '''

    :param order: int encoding fist second or third slice of the original input
    :return: function for slicing
    '''
    def _slicer(input):
      return input[:,784 * order : 784 * (order + 1)]
    return _slicer

  anchor = Reshape((28,28,1))(Lambda(_get_slicer_at(0))(concat_input))
  positive = Reshape((28,28,1))(Lambda(_get_slicer_at(1))(concat_input))
  negative = Reshape((28,28,1))(Lambda(_get_slicer_at(2))(concat_input))

  anchor_encoded = shared_model(anchor)
  positive_encoded = shared_model(positive)
  negative_encoded = shared_model(negative)

  predictions = Concatenate(axis=1)([anchor_encoded,positive_encoded,negative_encoded])
  
  model = Model(inputs=[concat_input],outputs=[predictions])
  model.compile(loss=triplet_loss_with_arg(12544,alpha=0.5),optimizer='adam')
  model.summary()

  print("starting training session")

  test_data_generator = MnistTripletGenerator(1000)

  model.fit_generator(
    generator=test_data_generator,
  )

  code.interact(local=locals())
  

if __name__=="__main__":
  test_network_main()