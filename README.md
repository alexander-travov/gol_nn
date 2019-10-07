CNN for Conway's Game of Life
=============================

Create, train and test CNN model for perfect prediction in the Game of Life.


Architecture
------------

Input datasets are wrapped to emulate cyclic nature of the game field.
Without that, you would get errors on the border.

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
(env) $ python gol.py 10 15

Building model:
2019-10-07 12:37:58.745726: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2494230000 Hz
2019-10-07 12:37:58.746573: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x38b9730 executing computations on platform Host. Devices:
2019-10-07 12:37:58.746674: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1_3x3 (Conv2D)           (None, 10, 15, 5)         50        
_________________________________________________________________
conv2_1x1 (Conv2D)           (None, 10, 15, 7)         42        
_________________________________________________________________
conv3_1x1 (Conv2D)           (None, 10, 15, 1)         8         
=================================================================
Total params: 100
Trainable params: 100
Non-trainable params: 0
_________________________________________________________________

Training model:
Train on 15000 samples, validate on 3000 samples
Epoch 1/8
15000/15000 [==============================] - 2s 163us/sample - loss: 0.1800 - accuracy: 0.7221 - val_loss: 0.1537 - val_accuracy: 0.7461
Epoch 2/8
15000/15000 [==============================] - 2s 105us/sample - loss: 0.1330 - accuracy: 0.8207 - val_loss: 0.1194 - val_accuracy: 0.8585
Epoch 3/8
15000/15000 [==============================] - 2s 106us/sample - loss: 0.1098 - accuracy: 0.8454 - val_loss: 0.0969 - val_accuracy: 0.8682
Epoch 4/8
15000/15000 [==============================] - 2s 104us/sample - loss: 0.0830 - accuracy: 0.9066 - val_loss: 0.0704 - val_accuracy: 0.9113
Epoch 5/8
15000/15000 [==============================] - 2s 108us/sample - loss: 0.0602 - accuracy: 0.9233 - val_loss: 0.0492 - val_accuracy: 0.9422
Epoch 6/8
15000/15000 [==============================] - 2s 106us/sample - loss: 0.0387 - accuracy: 0.9700 - val_loss: 0.0275 - val_accuracy: 0.9825
Epoch 7/8
15000/15000 [==============================] - 2s 111us/sample - loss: 0.0207 - accuracy: 0.9973 - val_loss: 0.0151 - val_accuracy: 1.0000
Epoch 8/8
15000/15000 [==============================] - 2s 104us/sample - loss: 0.0123 - accuracy: 1.0000 - val_loss: 0.0099 - val_accuracy: 1.0000

Evaluating model:
P_alive=0.1 Loss:0.01 Acc:1.00
P_alive=0.2 Loss:0.01 Acc:1.00
P_alive=0.3 Loss:0.00 Acc:1.00
P_alive=0.4 Loss:0.00 Acc:1.00
P_alive=0.5 Loss:0.00 Acc:1.00
P_alive=0.6 Loss:0.00 Acc:1.00
P_alive=0.7 Loss:0.00 Acc:1.00
P_alive=0.8 Loss:0.00 Acc:1.00
P_alive=0.9 Loss:0.00 Acc:1.00
```
