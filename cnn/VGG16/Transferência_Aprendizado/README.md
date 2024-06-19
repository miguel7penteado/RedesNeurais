# Transferência de Aprendizado (do inglês Transfer Learning)

A aprendizagem por transferência é uma das ferramentas mais úteis se você estiver trabalhando em qualquer tipo de problema de classificação de imagens. Mas o que é exatamente? Como você pode implementá-lo? Quão preciso é isso? Este artigo se aprofundará no aprendizado por transferência e mostrará como aplicá-lo usando a biblioteca Keras.

Observe que um pré-requisito para a aprendizagem por transferência de aprendizagem é ter conhecimento básico de redes neurais convolucionais (CNN), uma vez que a classificação de imagens exige o uso deste algoritmo.

As CNNs fazem uso de camadas de convolução que utilizam filtros para ajudar a reconhecer os recursos importantes de uma imagem. Esses recursos, que são muitos, ajudam a distinguir uma imagem específica. Sempre que você treina uma CNN em um monte de imagens, todos esses recursos são aprendidos internamente. Além disso, quando você usa uma CNN profunda, o número de parâmetros que estão sendo aprendidos – também chamados de pesos – pode chegar a milhões. Portanto, quando vários parâmetros precisam ser aprendidos, isso leva tempo. E é aqui que a aprendizagem por transferência ajuda.

[imagem]

Veja como.
Por exemplo, digamos que haja um problema chamado classificação ImageNet, um desafio popular de classificação de imagens onde existem milhões de imagens. Você precisa usar essas imagens para predizê-las e classificá-las em milhares de classes.

[imagem]

Fonte: [https://www.turing.com/kb/transfer-learning-using-cnn-vgg16](https://www.turing.com/kb/transfer-learning-using-cnn-vgg16)

