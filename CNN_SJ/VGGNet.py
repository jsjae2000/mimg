#### model construction
model = Sequential()

model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, img_rows, img_cols)))
model.add(PReLU())
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(PReLU())
model.add(MaxPooling2D( pool_size=(2, 2), stride=(2, 2) ))

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(PReLU())
model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(PReLU())
model.add(MaxPooling2D( pool_size=(2, 2), stride=(2, 2) ))

model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(PReLU())
model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(PReLU())
model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(PReLU())
model.add(MaxPooling2D( pool_size=(2, 2), stride=(2, 2) ))

model.add(Convolution2D(512, 3, 3, border_mode='same'))
model.add(PReLU())
model.add(Convolution2D(512, 3, 3, border_mode='same'))
model.add(PReLU())
model.add(Convolution2D(512, 3, 3, border_mode='same'))
model.add(PReLU())
model.add(MaxPooling2D( pool_size=(2, 2), stride=(2, 2) ))

model.add(Convolution2D(512, 3, 3, border_mode='same'))
model.add(PReLU())
model.add(Convolution2D(512, 3, 3, border_mode='same'))
model.add(PReLU())
model.add(Convolution2D(512, 3, 3, border_mode='same'))
model.add(PReLU())
model.add(MaxPooling2D( pool_size=(2, 2), stride=(2, 2) ))

model.add(Flatten())

model.add(Dense(4096))
model.add(PReLU())
model.add(Dropout(0.5))
model.add(Dense(4096))
model.add(PReLU())
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)

hist = model.fit(X_train, Y_train, batch_size=batch_size, 
                 nb_epoch=nb_epoch, show_accuracy=True, 
		 verbose=1, validation_split=0.2,
                 callbacks=[EarlyStopping(monitor='val_loss', patience=3),
                            ModelCheckpoint(filepath=_model_+'.h5', monitor='val_loss', save_best_only=True)])

