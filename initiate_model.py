from keras.layers import Dense, Input, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

# Initiate a model with provided parameters. This method simplifies the process of building new model structure
def initiate_model(conv_num, fc_num, conv_filters_num, conv_filters_size, pool_size, pool_pos, dropout, fc_denses_num, lr):
    output_classes = 7
    pic_size = 56

    model = Sequential()

    for i in range(conv_num):
        if i == 0:
            model.add(Conv2D(conv_filters_num[i],(conv_filters_size[i], conv_filters_size[i]), padding='same', input_shape=(pic_size, pic_size,1)))
        else:
            model.add(Conv2D(conv_filters_num[i],(conv_filters_size[i], conv_filters_size[i]), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        if i + 1 in pool_pos:
            model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
            model.add(Dropout(dropout))

    model.add(Flatten())

    for i in range(fc_num):
        model.add(Dense(fc_denses_num[i]))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

    model.add(Dense(output_classes, activation='softmax'))

    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model