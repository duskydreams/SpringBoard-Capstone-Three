import tensorflow as tf 
from tensorflow.keras import layers, models

def augementate(images, direction, rotation_factor, height_factor, width_factor):
    x = layers.RandomFlip(direction)(images)
    x = layers.RandomRotation(rotation_factor)(x)
    x = layers.RandomTranslation(height_factor, width_factor)(x)

    return x

def build_cnn (input_shape = (32,32,3), num_class = 10): 
    inputs = layers.Input(name = "input_image", shape = input_shape)

    pipeline_input = augementate(inputs, 'horizontal', rotation_factor= .05, height_factor= .08, width_factor= .08)

    pipeline_input = layers.Rescaling(1.0 / 255.0) (pipeline_input)

    pipeline_input = layers.Conv2D(32,(3,3), padding = 'same', activation='relu')(pipeline_input)
    pipeline_input = layers.BatchNormalization()(pipeline_input)
    pipeline_input = layers.Conv2D(32,(3,3), padding = 'same', activation= 'relu')(pipeline_input)
    pipeline_input = layers.MaxPool2D (pool_size= (2,2))(pipeline_input)

    pipeline_input = layers.Conv2D(128,(3,3), padding='same', activation='relu')(pipeline_input)
    pipeline_input = layers.BatchNormalization()(pipeline_input)

    pipeline_input = layers.GlobalAveragePooling2D()(pipeline_input)

    pipeline_input = layers.Dense(128, activation='relu')(pipeline_input)
    pipeline_input = layers.Dropout(.4)(pipeline_input)

    outputs = layers.Dense(num_class, activation = 'softmax', name = 'predictions')(pipeline_input)

    return models.Model(inputs, outputs = outputs, name = 'mfr2_cnn')


def compile_file_name (person_name, image_num): 
    person_name = person_name.strip()
    return f"{person_name}_000{image_num}.png"

def mask_encoding(mask_status): 
    return 1 if mask_status.strip() == 'mask' else 0