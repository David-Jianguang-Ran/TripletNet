# Plan of Action

remember this project is meant to explore triplet loss function and training image transformers

## things to implement 

- triplet data generator

- inception like cnn using keras

- triplet loss function 

- embedding similarity comparison function (just distance function for testing)

## style and other things?

- _no inheritance allowed!_ <= lets not be silly here, oo has its place just like functional

- keep in functional for the confusing bits (not strict)


## notes on trained networks

### Encoders

- encoder - first version of network, using maxpooling and 3 inception modules, outputing 4k vector

- encoderII - similar to above, outputing a smaller 256-D vector 

- encoderIII - similar to above but using average pooling, outputing 256-D vector
-- this network seem to have better performance compared to above 
-- due to less 'information loss' at average pooling layers