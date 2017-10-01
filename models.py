from keras.layers import Dense, Dropout, Input, BatchNormalization, Reshape, dot
from keras.models import Model
import keras.backend as K


def nlayer_net(input_features, hidden=(100,), compile_net=True):
    inp = Input((input_features,))
    x = inp
    for h in hidden:
        x = Dense(h, activation='relu')(x)
        x = Dropout(0.5)(x)
    out = Dense(6, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    if compile_net:
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
    return model


def naive(input_features, hidden=100):
    return nlayer_net(input_features, hidden=[hidden])
