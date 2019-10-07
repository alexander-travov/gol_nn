CNN for Conway's Game of Life
=============================

Create, train and test CNN model for perfect prediction in the Game of Life.


Architecture
------------

To prevent errors on the border input datasets are wrapped
to emulate cyclic nature of the game field.

Next state of a cell depends only on its neighbours.

Thats why 3x3 convolution is a reasonable choice for the first layer.
``valid`` padding is used for this layer to shrink input shape down
to the original size of the field.

Then 1x1 convolution layer with num\_channels of filters is used.

Finally, another 1x1 convolution with 1 output channel after sigmoid nonlinearity
predicts the probability for the cell to be alive in the next state.

In this architecture every cell is processed in the same way as others
and only takes into account the receptive field of its 3x3 neighbourhood.

```
    (N, H+2, W+2, 1)
           |
        conv1 (3,3)
           |
    (N, H, W, num_filters)
           |
        conv2 (1,1)
           |
    (N, H, W, num_channels)
           |
        conv3 (1,1)
           |
    (N, H, W, 1)
```

To estimate the inner complexity of the game, one can count total number
of different 3x3 patterns: 2^9 = 512.

You can get away with only 5 filters and 7 channels for a total of 100 trainable parameters
and still get perfect accuracy on the test set.


Preparing environment
---------------------

``` bash
python3 -m venv env
source env/bin/activate
pip install -U pip setuptools
pip install -r requirements.txt
```


Usage
-----

Script ``gol.py`` trains and tests network for a given field size:

``` bash
$ python gol.py HEIGHT WIDTH
Building model
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 20, 30, 10)        100       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 20, 30, 20)        220       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 30, 1)         21        
=================================================================
Total params: 341
Trainable params: 341
Non-trainable params: 0
_________________________________________________________________

Training model
Train on 15000 samples, validate on 3000 samples
Epoch 1/5
15000/15000 [==============================] - 5s 314us/sample - loss: 0.5993 - accuracy: 0.6937 - val_loss: 0.4855 - val_accuracy: 0.7484
Epoch 2/5
15000/15000 [==============================] - 4s 265us/sample - loss: 0.2842 - accuracy: 0.9055 - val_loss: 0.1149 - val_accuracy: 1.0000
Epoch 3/5
15000/15000 [==============================] - 4s 262us/sample - loss: 0.0539 - accuracy: 1.0000 - val_loss: 0.0223 - val_accuracy: 1.0000
Epoch 4/5
15000/15000 [==============================] - 4s 263us/sample - loss: 0.0132 - accuracy: 1.0000 - val_loss: 0.0077 - val_accuracy: 1.0000
Epoch 5/5
15000/15000 [==============================] - 4s 263us/sample - loss: 0.0053 - accuracy: 1.0000 - val_loss: 0.0037 - val_accuracy: 1.0000

Evaluating model
P=0.10 Loss:0.00 Acc:1.00
P=0.20 Loss:0.00 Acc:1.00
P=0.30 Loss:0.00 Acc:1.00
P=0.40 Loss:0.00 Acc:1.00
P=0.50 Loss:0.00 Acc:1.00
P=0.60 Loss:0.00 Acc:1.00
P=0.70 Loss:0.00 Acc:1.00
P=0.80 Loss:0.00 Acc:1.00
P=0.90 Loss:0.00 Acc:1.00
```
