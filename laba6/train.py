# train.py
import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ====== Настройки ======
DATA_DIR = 'dataset'       # папка с подпапками классов
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'gesture_rps.h5')
LABELS_PATH = os.path.join(MODEL_DIR, 'labels.json')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

os.makedirs(MODEL_DIR, exist_ok=True)

# ====== Генераторы данных с аугментацией ======
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2)
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training', shuffle=True
)
val_gen = train_datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=False
)

num_classes = len(train_gen.class_indices)
print("Class indices:", train_gen.class_indices)

# ====== Модель (Transfer Learning MobileNetV2) ======
base = MobileNetV2(include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), weights='imagenet')
base.trainable = False  # на первом этапе — не дообучаем

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ====== Callbacks ======
callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# ====== Тренировка ======
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

# ====== Сохраняем labels ======
with open(LABELS_PATH, 'w', encoding='utf-8') as f:
    json.dump(train_gen.class_indices, f, ensure_ascii=False, indent=2)

print("Done. Model saved to:", MODEL_PATH)
