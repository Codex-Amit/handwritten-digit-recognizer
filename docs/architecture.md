# CNN Architecture

## Overview

The model uses a two-block convolutional backbone followed by a fully connected classifier.

## Layer Details

| # | Layer           | Output Shape     | Params  |
|---|----------------|-----------------|---------|
| 1 | Input           | (28, 28, 1)      | 0       |
| 2 | Conv2D 32×3×3   | (28, 28, 32)     | 320     |
| 3 | BatchNorm       | (28, 28, 32)     | 128     |
| 4 | Conv2D 32×3×3   | (28, 28, 32)     | 9,248   |
| 5 | BatchNorm       | (28, 28, 32)     | 128     |
| 6 | MaxPool 2×2     | (14, 14, 32)     | 0       |
| 7 | Dropout 0.25    | (14, 14, 32)     | 0       |
| 8 | Conv2D 64×3×3   | (14, 14, 64)     | 18,496  |
| 9 | BatchNorm       | (14, 14, 64)     | 256     |
|10 | Conv2D 64×3×3   | (14, 14, 64)     | 36,928  |
|11 | BatchNorm       | (14, 14, 64)     | 256     |
|12 | MaxPool 2×2     | (7, 7, 64)       | 0       |
|13 | Dropout 0.25    | (7, 7, 64)       | 0       |
|14 | Flatten         | (3136,)          | 0       |
|15 | Dense 128       | (128,)           | 401,536 |
|16 | BatchNorm       | (128,)           | 512     |
|17 | Dropout 0.5     | (128,)           | 0       |
|18 | Dense 10        | (10,)            | 1,290   |

**Total trainable parameters: ~468K**

## Design Decisions

- **Double convolution blocks** improve feature extraction vs single conv layers.
- **BatchNormalization** after each conv speeds up training and acts as regularization.
- **Dropout** prevents overfitting: 0.25 after pooling, 0.5 before output.
- **Adam optimizer** with `ReduceLROnPlateau` for adaptive learning rate decay.
