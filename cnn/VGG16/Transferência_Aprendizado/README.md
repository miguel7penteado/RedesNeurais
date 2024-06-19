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

## Implementando aprendizagem por transferência

Agora que o conjunto de dados foi carregado, é hora de implementar a aprendizagem por transferência.
Comece importando VGG16 de keras.applications e forneça o tamanho da imagem de entrada. Os pesos são importados diretamente do problema de classificação ImageNet. Quando top = False, significa descartar os pesos da camada de entrada e da camada de saída, pois você usará suas próprias entradas e saídas.

```python
VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet',vgg include_top=False)
```

![](fotos/saida01.avif)

Para camada em vgg.layers, layer.trainable=False para indicar que todas as camadas no modelo VGG16 não devem ser treinadas novamente. Você deseja usar apenas este parâmetro diretamente.

```python
vgg.input
```

Saída:

![](fotos/saida02.avif)













Fonte: [https://www.turing.com/kb/transfer-learning-using-cnn-vgg16](https://www.turing.com/kb/transfer-learning-using-cnn-vgg16)

