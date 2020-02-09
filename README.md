# Deep-Learning-with-Keras

Las dos arquitecturas de red neuronal que actualmente se asocian al deep learning son las de convolución y las recurrentes. Las primeras tienen aplicación directa en el dominio del reconocimiento de imágenes, entre otros. Las segundas se aplican en dominios en los que la naturaleza de los datos es secuencial como, por ejemplo, procesamiento del habla y lenguaje escrito. 

Keras está basado en la idea de grafo (al igual que otros frameworks como, por ejemplo Tensorflow) en el que la estructura general de computación de los datos, desde que se introducen a la entrada, hasta que se genera una salida, van sufriendo transformaciones a partir de operaciones cuyo orden de ejecución viene determinado por dicho grafo en el que los nodos son operaciones y las aristas son caminos por los que pueden fluir los datos.

Keras es un wrapper, desarrollado en Python, a varios frameworks de deep learning de bajo nivel, incluyendo TensorFlow (al que está instanciado por defecto).

El concepto principal en Keras es el de modelo (red neuronal). El principal tipo de modelo es el ![secuencial](http://keras.io/getting-started/sequential-model-guide/), que se define como una disposición lineal de capas. Para crear un modelo se necesita especificar las capas que va a tener y el orden en el que se disponen.

Procesar por completo un modelo Keras implica:

* Especificar la forma de la entrada al modelo: cuántos atributos de entrada y el tamaño de los batches que vamos a usar en el entrenamiento.

* Compilar el modelo: se refiere a, primeramente, crear nuestro modelo para dejarlo listo para el entrenamiento. Seguidamente, hay que configurar el proceso de aprendizaje que se va a ejecutar sobre el mismo. Ahí han de especificarse, normalmente, tres elementos: el algoritmo de optimización de los parámetros del modelo, la función de error de la que va a hacer uso el algoritmo de optimización y las métricas a usar para medir el progreso del entrenamiento.

* Entrenar el modelo: mediante la función `fit` invocamos el entrenamiento.

### KERAS en R

En este proyecto vamos a usar Keras en el lenguaje R. Keras es una API que puede integrar diferentes motores de deep learning, entre ellos TensorFlow (TF). TF está programado en C++ y Python; Keras es 100% Python pero incorpora Wrappers que lo hacen accesible a otros lenguajes, por ejemplo R.

Para instalarlo, se aconseja usar Linux o Mac aunque también sería posible hacerlo funcionar en Windows. Toca instalar Keras en R. Las instrucciones pueden observarse en <https://keras.rstudio.com/>, pero básicamente, desde la consola R de Rstudio:

```{r, eval=FALSE}
library(keras)
install_keras(tensorflow="1.5")
```
## Objetivo del proyecto

En este proyecto nos vamos a centrar en evaluar las ventajas de una red de convolución con respecto a una red MLP (preferiblemente del paquete RSNNS) convencional (ya sea de una o más capas), para la clasificación de imágenes en el conjunto CIFAR-10 (<https://www.cs.toronto.edu/~kriz/cifar.html>). Este conjunto de imágenes, integrado en Keras, tiene un tamaño de 60000 × 32 × 32 × 3, i.e. 60000 imágenes, de tamaño 32 × 32 y tres canales RGB para el color, con valores en [0, 255].

Trabajaremos desde Keras con TensorFlow en el backend, con una métrica basada en el accuracy y el error de entropía cruzada, estimando ambos mediante una evaluación cruzada con k = 5. Por tanto, construiremos las siguientes redes:

* La construcción de una red MLP de arquitectura libre en la que se trabaje con un algoritmo de optimización adecuado y un control del overfitting apropiado dado el problema al que nos enfrentamos.

* La construcción de una red de convolución de arquitectura libre que supere en accuracy a la red MLP construida previamente. Para ello se deberá cuantificar el número total de filtros por capa, el tamaño de los mapas de activación generados y el número de pesos.

En la carpeta R/ se encuentran los scripts desarrollados para la realización de este proyecto, donde los ficheros Conv-Final.R y MLP-Final.R contienen los modelos finales de ambas redes que se obtienen del proceso de elección desarrollado en Conv-Selection.R y MLP-Selection.R.

Además, en la carpeta inst/ disponemos de una memoria en forma de fichero .Rmd donde se explica el proceso seguido para la realización del proyecto
