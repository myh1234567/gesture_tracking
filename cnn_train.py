from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

model= Sequential()

model.add(Convolution2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(200, 200,1)))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#Now generate training and test sets from folders

train_datagen=ImageDataGenerator(
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.,
                                   horizontal_flip = False
                                 )

test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory("Dataset/training_set",
                                               target_size = (200,200),
                                               color_mode='grayscale',
                                               batch_size=10,
                                               class_mode='categorical')

test_set=test_datagen.flow_from_directory("Dataset/test_set",
                                               target_size = (200,200),
                                               color_mode='grayscale',
                                               batch_size=10,
                                               class_mode='categorical')

# start to train
model.fit_generator(training_set,
                         samples_per_epoch = 3000,
                         nb_epoch = 10,
                         validation_data = test_set,
                         nb_val_samples = 320)

model.save_weights("weights.hdf5",overwrite=True)
model_json = model.to_json()
with open("model.json", "w") as model_file:
    model_file.write(model_json)
print("Finished.")






