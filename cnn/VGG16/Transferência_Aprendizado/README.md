# Transferência de Aprendizado (do inglês Transfer Learning)

A aprendizagem por transferência é uma das ferramentas mais úteis se você estiver trabalhando em qualquer tipo de problema de classificação de imagens. Mas o que é exatamente? Como você pode implementá-lo? Quão preciso é isso? Este artigo se aprofundará no aprendizado por transferência e mostrará como aplicá-lo usando a biblioteca Keras.

Observe que um pré-requisito para a aprendizagem por transferência de aprendizagem é ter conhecimento básico de redes neurais convolucionais (CNN), uma vez que a classificação de imagens exige o uso deste algoritmo.

As CNNs fazem uso de camadas de convolução que utilizam filtros para ajudar a reconhecer os recursos importantes de uma imagem. Esses recursos, que são muitos, ajudam a distinguir uma imagem específica. Sempre que você treina uma CNN em um monte de imagens, todos esses recursos são aprendidos internamente. Além disso, quando você usa uma CNN profunda, o número de parâmetros que estão sendo aprendidos – também chamados de pesos – pode chegar a milhões. Portanto, quando vários parâmetros precisam ser aprendidos, isso leva tempo. E é aqui que a aprendizagem por transferência ajuda.

![](fotos/imagem01.avif)

Veja como.
Por exemplo, digamos que haja um problema chamado classificação ImageNet, um desafio popular de classificação de imagens onde existem milhões de imagens. Você precisa usar essas imagens para predizê-las e classificá-las em milhares de classes.

![](fotos/imagem02.avif)

Todos os anos, um modelo supera o outro. Uma vez estabelecido que um determinado modelo tem o melhor desempenho, todos os parâmetros ou todos os pesos que ele aprendeu são disponibilizados publicamente. Usando o aplicativo Keras, você pode usar diretamente o melhor modelo e todos os pesos pré-treinados para não precisar executar o processo de treinamento novamente. Isso economiza muito tempo.

A lista de todos os modelos disponíveis pode ser encontrada no sitePágina de documentação do Keras. Eles foram treinados no problema de classificação ImageNet e podem ser usados ​​diretamente.

A precisão desses modelos, bem como os parâmetros que eles usaram, podem ser vistos. A profundidade dos modelos também é mostrada.
Aqui está uma olhada no código.

## Carregando conjunto de dados

Primeiro, importe as bibliotecas necessárias.

```python

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

```

A seguir, mencione o tamanho da imagem. Lembre-se de que o modelo foi treinado no problema de classificação ImageNet, portanto pode ter um tamanho de entrada diferente. Como o problema e a imagem específica podem ter tamanhos diferentes, é necessário alterar a camada de entrada.

```python
IMAGE_SIZE = [224, 224]
```

O problema de classificação do ImageNet tem como saída 1.000 classes, mas você também pode ter menos. Saiba que você precisa fazer uma alteração na camada de saída. Todas as camadas ocultas e todas as camadas de convolução e os pesos dessas camadas permanecem os mesmos.

Para fins de demonstração, usaremos o conjunto de dados de câncer de pele, que contém uma série de imagens classificadas como benignas e malignas. O conjunto de dados pode ser baixado [aqui](https://www.kaggle.com/fanconic/skin-cancer-malignant-vs-benign)

Na próxima etapa, especifique o caminho do trem e o caminho do teste.

```python

train_path = '/content/drive/MyDrive/skincancerdataset/train'
test_path  = '/content/drive/MyDrive/skincancerdataset/test'

```

```python

from PIL import Image
import os
from IPython.display import display
from IPython.display import Image as _Imgdis
folder = train_path+ '/benign'

```

Este próximo passo, que não é obrigatório, exibe as imagens benignas.

Saída 01:

![](fotos/amostra01.avif)

Saída 02:

![](fotos/amostra02.avif)


## Implementando aprendizagem por transferência

Agora que o conjunto de dados foi carregado, é hora de implementar a aprendizagem por transferência.
Comece importando VGG16 de keras.applications e forneça o tamanho da imagem de entrada. Os pesos são importados diretamente do problema de classificação ImageNet. Quando top = False, significa descartar os pesos da camada de entrada e da camada de saída, pois você usará suas próprias entradas e saídas.

```python
VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet',vgg include_top=False)
```

Saída:

![](fotos/saida01.avif)

Para camada em vgg.layers, layer.trainable=False para indicar que todas as camadas no modelo VGG16 não devem ser treinadas novamente. Você deseja usar apenas este parâmetro diretamente.

```python
vgg.input
```

Saída:

![](fotos/saida02.avif)

```python
for layer in vgg.layers:
    layer.trainable = False
```

```python
folders glob('/content/drive/MyDrive/skincancerdataset/train/*')
print(len(folders))
```

Você pode obter o número de pastas usando glob.

Saída:

![](fotos/saida03.avif)

Em seguida, especifique uma camada nivelada para que qualquer saída obtida na última camada seja condensada em uma dimensão. Você precisa de uma camada de saída com apenas dois neurônios. A função de ativação utilizada é softmax. Você também pode usar sigmoid, pois a saída possui apenas duas classes, mas esta é a forma mais generalizada.

```python

x = Flatten()(vgg.output)
prediction Dense(len(folders), activation='softmax') (x)
model = Model(inputs=vgg.input, outputs prediction)
model.summary()

```

Saída:

![](fotos/saida04.avif)

![](fotos/saida05.avif)

Agora você pode testemunhar a magia da aprendizagem por transferência. O total de parâmetros é de 14 milhões, mas como você pode ver, os parâmetros treináveis ​​são de apenas 15.000. Isso reduz muito tempo e elimina grande parte da complexidade.

A etapa seguinte experimenta o otimizador Adam, a função de perda binária_crossentropy e a precisão como métricas.

```python

from keras import optimizers

adam = optimizers.Adam()
model.compile(loss='binary_crossentropy', optimizer adam, metrics=['accuracy'])

```

## Aumento de dados

A próxima etapa é o aumento da imagem. Você importará prepocess_input pois houve algumas etapas de pré-processamento quando o modelo real foi treinado no problema imagenet. Para obter resultados semelhantes, você precisa usar as etapas exatas de pré-processamento. Alguns, incluindo deslocamento e zoom, são usados ​​para reduzir o overfitting.


```python

train_datagen = ImageDataGenerator( preprocessing_function=preprocess_input, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,

```

O mesmo é feito para o conjunto de testes.

```python

test_datagen = ImageDataGenerator( preprocessing_function=preprocess_input, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,

```

Especifique o tamanho de destino da saída, o tamanho do lote e a classe.

```python

train_set = train_datagen.flow_from_directory(train_path, target_size = (224, 224), batch_size = 32, class_mode 'categorical')

```

Saída:

![](fotos/saida06.avif)

O mesmo é feito para o conjunto de testes.

```python

test_set = test_datagen.flow_from_directory(test_path, target_size = (224, 224), batch_size = 32, class_mode = categorical')

```

Saída:

![](fotos/saida07.avif)

## Treinando o modelo

Agora que o aumento de dados foi concluído, é hora de treinar o modelo. O ponto de verificação do modelo é usado para salvar o melhor modelo. Você usará 10 épocas com 5 etapas por época. As etapas de validação são iguais a 32.

```python
from datetime import datetime
from keras.callbacks import ModelCheckpoint

checkpoint ModelCheckpoint(filepath='mymodel.h5', verbose=2, save_best_only=True) 

callbacks = [checkpoint]

```

Saída:

![](fotos/saida08.avif)

![](fotos/saida09.avif)


```python

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Precição dos valores do modelo CNN')
plt.ylabel('Precição')
plt.xlabel('Epoch')

```

Saída:

![](fotos/grafico01.avif)

A precisão pode ser vista assim que o treinamento for concluído. A melhor parte é que você realmente não precisa fazer nada, exceto pegar os pesos diretamente e desenvolver o modelo de melhor desempenho no conjunto de dados popular.

## Vantagens de usar o aprendizado de transferência no aprendizado de máquina

- Isso economiza tempo e recursos. A maioria dos problemas de aprendizado de máquina envolve o treinamento de uma grande quantidade de dados. Esse tipo de dados de treinamento rotulados leva mais tempo. No entanto, na aprendizagem por transferência a maioria dos modelos são pré-treinados, o que reduz o tamanho dos dados de treinamento.

- Melhora a eficiência de um modelo durante o treinamento. O desenvolvimento de modelos de aprendizado de máquina para resolver problemas complexos é demorado. Com a aprendizagem por transferência, você não precisa criar um modelo do zero. Você pode reutilizar o modelo desenvolvido transferindo seu conhecimento.

- Em vez de usar algoritmos diferentes para resolver novos problemas, a aprendizagem por transferência fornece uma maneira mais generalizada de resolver o problema.

## Aplicações de aprendizagem por transferência

Aqui estão algumas aplicações reais de aprendizagem por transferência.

- No processamento de linguagem natural, a aprendizagem por transferência pode ser usada para prever a próxima palavra em uma sequência.

- Os modelos de aprendizagem por transferência são adequados para reconhecer imagens. Por exemplo, um modelo desenvolvido para identificar gatos pode ser usado para identificar cães.

- No reconhecimento de fala, o modelo desenvolvido para reconhecer um idioma pode ser usado para reconhecer outro idioma.

- O modelo desenvolvido para reconhecer exames de ressonância magnética também pode ser usado para detectar exames de tomografia computadorizada.

- Um modelo de aprendizado de máquina desenvolvido para classificar e-mails pode ser usado para verificar e-mails de spam.

Conforme demonstrado, a aprendizagem por transferência é uma técnica muito eficaz quando se trabalha em problemas de classificação de imagens. Agora que você aprendeu usando a CNN, você pode experimentar diferentes modelos e realizar o ajuste de hiperparâmetros usando o sintonizador Keras.



Fonte: [https://www.turing.com/kb/transfer-learning-using-cnn-vgg16](https://www.turing.com/kb/transfer-learning-using-cnn-vgg16)

