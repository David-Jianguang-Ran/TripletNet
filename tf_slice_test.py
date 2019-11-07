import tensorflow
import code

my_tensor = tensorflow.constant([
  [1,2,3,4],
  [2,2,3,4],
  [3,2,3,4],
  [4,2,3,4]
])

split = tensorflow.slice(my_tensor,[0,0],[4,2])
# result
# array([[1, 2],
#        [2, 2],
#        [3, 2],
#        [4, 2]], dtype=int32)

tensorflow.print(split)

sess = tensorflow.Session()

code.interact(local=locals())