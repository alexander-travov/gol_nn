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
(env) $ python gol.py 20 30
Building model:
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv1_3x3 (Conv2D)           (None, 20, 30, 5)         50        
_________________________________________________________________
conv2_1x1 (Conv2D)           (None, 20, 30, 7)         42        
_________________________________________________________________
conv3_1x1 (Conv2D)           (None, 20, 30, 1)         8         
=================================================================
Total params: 100
Trainable params: 100
Non-trainable params: 0
_________________________________________________________________

Training model:
Train on 8000 samples, validate on 2000 samples
Epoch 1/2
8000/8000 [==============================] - 2s 273us/sample - loss: 0.1353 - accuracy: 0.8157 - val_loss: 0.0522 - val_accuracy: 0.9863
Epoch 2/2
8000/8000 [==============================] - 2s 197us/sample - loss: 0.0168 - accuracy: 0.9980 - val_loss: 0.0034 - val_accuracy: 1.0000

Evaluating model:
P_alive=0.1 Loss:0.00 Acc:1.00
P_alive=0.2 Loss:0.01 Acc:1.00
P_alive=0.3 Loss:0.01 Acc:1.00
P_alive=0.4 Loss:0.01 Acc:1.00
P_alive=0.5 Loss:0.00 Acc:1.00
P_alive=0.6 Loss:0.00 Acc:1.00
P_alive=0.7 Loss:0.00 Acc:1.00
P_alive=0.8 Loss:0.00 Acc:1.00
P_alive=0.9 Loss:0.00 Acc:1.00
```
