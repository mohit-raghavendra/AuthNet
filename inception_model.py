
print("Called Preprocessing")
x_train, x_val, y_train, y_val = preprocess_data(data_df)


def build_model_Inception():

    inception = inception_v3.InceptionV3(
        include_top=False,
        input_shape=(224,224,3)
    )

    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
    

    model = Sequential()
    model.add(inception)
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(number_of_people, activation='softmax'))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=0.0001),
        metrics=['accuracy']
    )
    
    return model

model = build_model_Inception()
model.summary()


BATCH_SIZE = 32

model.fit()
model.predict()