
```python 

import tensorflow as tf

BATCH_SIZE = 64

flowers = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
)

train_ds = img_gen.flow_from_directory(flowers, batch_size=BATCH_SIZE, shuffle=True, class_mode='sparse')
num_classes = 5

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(256, 256, 3)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

epochs=10
history = model.fit(
  train_ds,
  epochs=epochs
)

```

Se você agrupar flow_from_directory com tf.data.Dataset.from_generator assim:


```python

train_ds = tf.data.Dataset.from_generator(
    lambda: img_gen.flow_from_directory(flowers, batch_size=BATCH_SIZE, shuffle=True, class_mode='sparse'),
    output_types=(tf.float32, tf.float32))
```

Você notará que a barra de progresso se parece com isto porque steps_per_epoch não foi definido explicitamente:

```bash
Epoch 1/10
Found 3670 images belonging to 5 classes.
     29/Unknown - 104s 4s/step - loss: 2.0364
```	    

E se você adicionar este parâmetro, verá as etapas na barra de progresso:

```python
history = model.fit(
  train_ds,
  steps_per_epoch = len(from_directory),
  epochs=epochs
)
```

```bash
Found 3670 images belonging to 5 classes.
Epoch 1/10
 3/58 [>.............................] - ETA: 3:19 - loss: 4.1357
```

Como usar este gerador corretamente com função adequada para ter todos os dados em meu conjunto de treinamento, incluindo imagens originais, não aumentadas e imagens aumentadas, e percorrê-los várias vezes/passo?

Você pode simplesmente aumentar o steps_per_epoch além do número de amostras // batch_size multiplicando por algum fator:

```python
history = model.fit(
  train_ds,
  steps_per_epoch = len(from_directory)*2,
  epochs=epochs
)
```

```bash
Found 3670 images belonging to 5 classes.
Epoch 1/10
  1/116 [..............................] - ETA: 12:11 - loss: 1.5885
```

Agora, em vez de 58 passos por época, você tem 116.

