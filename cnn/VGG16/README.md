
%tensorflow_version 2.x

import tensorflow as tf
import cv2
import numpy as np
import os

import keras

from keras.preprocessing.Image import ImageDataGenerator
from keras.layers              import Dense,Flatten,Conv2D,Activation,Dropout
from keras                     import backend as K
from keras.models              import Sequential, Model
from keras.models              import load_model
from keras.optimizers          import SGD
from keras.callbacks           import EarlyStopping,ModelCheckpoint
from keras.layers              import MaxPool2D
from google.colab.patches      import cv2_imshow

'''
O objetivo do ImageDataGenerator é importar facilmente dados com rótulos para o modelo. 
É uma classe muito útil, pois possui muitas funções para redimensionar, girar, aplicar zoom, inverter, etc. 
O mais útil dessa classe é que ela não afeta os dados armazenados no disco. 
Esta classe altera os dados em movimento enquanto os passa para o modelo. 
O ImageDataGenerator rotulará automaticamente todos os dados dentro da pasta. 
Dessa forma, os dados ficam facilmente prontos para serem transmitidos à rede neural.
'''

train_datagen   = ImageDataGenerator (zoom_range=0,15,width_shift_range=0,2,height_shift_range=0,2,shear_range=0,15)
test_datagen    = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(“/content/drive/MyDrive/Rohini_Capstone/Car Images/Train Images”,target_size=(224, 224),batch_size=32,shuffle=True,class_mode='categorical')
test_generator  = test_datagen.flow_from_directory(“/content/drive/MyDrive/Rohini_Capstone/Car Images/Test Images/”,target_size=(224.224),batch_size=32,shuffle=False,class_mode='categorical')


'''
Defina o modelo VGG16 como modelo sequencial

→ 2 x camada de convolução de  64 canais de kernel 3x3 e mesmo preenchimento
→ 1 x camada    maxpool    de tamanho    de pool   2x2 e passo 2x2
→ 2 x camada de convolução de 128 canais de kernel 3x3 e mesmo preenchimento
→ 1 x camada    maxpool    de tamanho    de pool   2x2 e passo 2x2
→ 3 x camada de convolução de 256 canais de kernel 3x3 e mesmo preenchimento
→ 1 x camada    maxpool    de tamanho    de pool   2x2 e passo 2x2
→ 3 x camada de convolução de 512 canais de kernel 3x3 e mesmo preenchimento
→ 1 x camada    maxpool    de tamanho    de pool   2x2 e passo 2x2
→ 3 x camada de convolução de 512 canais de kernel 3x3 e mesmo preenchimento
→ 1 x camada    maxpool    de tamanho    de pool   2x2 e passo 2x2

Também adicione a ativação ReLu (Unidade Linear Retificada) 
a cada camada para que 
todos os valores negativos NÃO SEJEM passados ​​para a próxima camada.
'''

def VGG16():

model = Sequential()

# (224x224 pixels em 3 filtros de cores)/1 => (224x224 pixels em 64 filtros de cores)
# 2 cadamas de 64 filtros (kernel 3x3) mesmo preenchimento passada 1 + 1 camada maxpool 2x2 passada 2
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding=”same”, activation=”relu”))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding=”same”, activation=”relu”))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# (112x112 pixels em 128 filtros de cores) => (224x224/2)
# 2 cadamas de 128 filtros (kernel 3x3) mesmo preenchimento passada 1 + 1 camada maxpool 2x2 passada 2
model.add(Conv2D(filters=128, kernel_size=(3,3), padding=”same”, activation=”relu”))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding=”same”, activation=”relu”))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# (56x56 pixels em 256 filtros de cores) => (224x224/4)
# 3 cadamas de 256 filtros (kernel 3x3) mesmo preenchimento passada 1 + 1 camada maxpool 2x2 passada 2
model.add(Conv2D(filters=256, kernel_size=(3,3), padding=”same”, activation=”relu”))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding=”same”, activation=”relu”))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding=”same”, activation=”relu”))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# (28x28 pixels em 512 filtros de cores) => (224x224/8)
# 3 cadamas de 512 filtros (kernel 3x3) mesmo preenchimento passada 1 + 1 camada maxpool 2x2 passada 2
model.add(Conv2D(filters=512, kernel_size=(3,3), padding=”same”, activation=”relu”))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding=”same”, activation=”relu”))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding=”same”, activation=”relu”))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# (14x14 pixels em 512 filtros de cores) => (224x224/16)
# 3 cadamas de 512 filtros (kernel 3x3) mesmo preenchimento passada 1 + 1 camada maxpool 2x2 passada 2
model.add(Conv2D(filters=512, kernel_size=(3,3), padding=”same”, activation=”relu”))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding=”same”, activation=”relu”))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding=”same”, activation=”relu”))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name=’vgg16'))
# (7x7 pixels em 512 filtros de cores) => (224x224/32)


#Três camadas totalmente conectadas (FC) seguem uma pilha de camadas convolucionais: 
model.add(Flatten(name=’flatten’))

# (1x1 pixels em 4.096 filtros de cores) 
# as duas primeiras têm 4.096 canais cada
model.add(Dense(256, activation=’relu’, name=’fc1'))
model.add(Dense(128, activation=’relu’, name=’fc2'))

# a terceira realiza a 
# classificação ILSVRC de 1.000 vias (ImageNet Large Scale Visual Recognition Challenge) e, 
# portanto, contém 1.000 canais (um para cada classe). 
# OBS: a camada final é a camada soft-max.

# (1x1 pixels em 1.000 filtros de cores)
model.add(Dense(196, activation=’softmax’, name=’output’))

return model

'''
Após criar toda a convolução, passe os dados para a camada densa para isso achatamos o vetor que sai das convoluções e somamos:

→ 1 x Camada Densa de 256 unidades
→ 1 x Camada Densa de 128 unidades
→ 1 x camada Dense Softmax de 2 unidades

Altere a dimensão da camada de saída para o número de classes na linha abaixo. 
O conjunto de dados de Stanford possui 196 classes e, portanto, o mesmo é mencionado na camada de saída. 
Além disso, a função de ativação Softmax é usada porque é um algoritmo de classificação de objetos. 
A camada softmax produzirá o valor entre 0 e 1 com base na confiança do modelo à qual classe as imagens pertencem.
'''

model.add(Dense(196, activation=’softmax’, name=’output’))

build the model and print the summary to know the structure of the model.

model=VGG16()

model.summary()

Vgg16 = Model(inputs=model.input, outputs=model.get_layer(‘vgg16’).output)

Load the pre-trained weights of the VGG16 model (gg16_weights_tf_dim_ordering_tf_kernels_notop.h5)so that we don’t have to retrain the entire model.

Vgg16.load_weights(“/content/drive/MyDrive/Rohini_Capstone/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5”)

'''
Defina a parada antecipada, para que possamos interromper o treinamento se a precisão do modelo atingir o máximo com a das iterações anteriores.
'''

es=EarlyStopping(monitor=’val_accuracy’, mode=’max’, verbose=1, patience=20)

'''
Usei o SGD (descida gradiente estocástica) como otimizador (também podemos usar ADAM) como otimizador. 
Perda “A entropia cruzada categórica foi usada por ser um modelo de 196 classes. 
Podemos usar entropia cruzada binária se houver apenas duas classes. 
Especificamos a taxa de aprendizagem do otimizador; aqui, neste caso, está definido como 1e-6. 
Se o nosso treinamento está oscilando muito ao longo das épocas, então precisamos diminuir a taxa de aprendizagem para que possamos atingir os mínimos globais.

Compile o modelo, o otimizador e as métricas de perda.
'''

opt = SGD(learning_rate=1e-6, momentum=0.9)

model.compile(loss=”categorical_crossentropy”, optimizer=opt,metrics=[“accuracy”])

'''
O código abaixo não treinará o modelo VGG16 já treinado para que possamos usar os pesos pré-treinados para classificação. 
Isso é chamado de aprendizagem por transferência, que é usado para economizar muito esforço e recursos para retreinamento.

para camada em Vgg16.layers:
'''

for layer in Vgg16.layers:

layer.trainable = False

for layer in model.layers:

print(layer, layer.trainable)

'''
O ponto de verificação do modelo é criado e podemos salvar apenas o melhor modelo para que possa ser usado novamente para teste e validação do modelo.
'''

mc = ModelCheckpoint(‘/content/drive/MyDrive/Rohini_Capstone/vgg16_best_model_1.h5’, monitor=’val_accuracy’, mode=’max’, save_best_only=True)

'''
ajustar o modelo ao gerador de dados de treinamento e teste. Executamos o modelo por 150 épocas.
'''

H = model.fit_generator(train_generator,validation_data=test_generator,epochs=150,verbose=1,callbacks=[mc,es])


'''
A precisão de 60% foi alcançada e o modelo foi interrompido devido ao aprendizado precoce em 131 épocas.

Depois que o modelo for treinado, também podemos visualizar a precisão do treinamento/validação
'''

import matplotlib.pyplot as plt
plt.plot(model.history[“acc”])
plt.plot(model.history[‘val_acc’])
plt.plot(model.history[‘loss’])
plt.plot(hist.history[‘val_loss’])
plt.title(“model accuracy”)
plt.ylabel(“Accuracy”)
plt.xlabel(“Epoch”)
plt.legend([“Accuracy”,”Validation Accuracy”,”loss”,”Validation Loss”])
plt.show()

'''
Para fazer previsões no modelo treinado, preciso carregar o melhor modelo salvo, pré-processar a imagem e passá-la ao modelo para saída.

Use o modelo salvo e carregue-os para avaliar o modelo.
'''

model.load_weights(“/content/drive/MyDrive/Rohini_Capstone/vgg16_best_model_1.h5”)

model.evaluate_generator(test_generator)

model_json = model.to_json()

with open(“/content/drive/MyDrive/Rohini_Capstone/vgg16_cars_model.json”,”w”) as json_file:

json_file.write(model_json)

from keras.models import model_from_json

run the prediction

def predict_(image_path):

#Carregue o modelo do arquivo JSON

json_file = open(‘/content/drive/MyDrive/Rohini_Capstone/vgg16_cars_model.json’, ‘r’)

model_json_c = json_file.read()

json_file.close()

model_c = model_from_json(model_json_c)

#Carregue os pesos

model_c.load_weights(“/content/drive/MyDrive/Rohini_Capstone/vgg16_best_model.h5”)

# Compile o modelo

opt = SGD(lr=1e-4, momentum=0.9)

model_c.compile(loss=”categorical_crossentropy”, optimizer=opt,metrics=[“accuracy”])

# Carregue a imagem que você deseja classificar

image = cv2.imread(image_path)

image = cv2.resize(image, (224,224))

cv2_imshow(image)

#Prediga a imagem

preds = model_c.predict_classes(np.expand_dims(image, axis=0))[0]

print(“Predicted Label”,preds)

predict_(“/content/drive/MyDrive/Rohini_Capstone/Car Images/Test Images/Rolls-Royce Phantom Sedan 2012/06155.jpg”)

