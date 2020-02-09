library(tensorflow)
library(keras)
library(caret)

set.seed(12345)
dswsize = 10000
cifar10 <- dataset_cifar10()
dssize = 50000
mask = sample (1:dssize,dswsize)
cifar10$train$x = cifar10$train$x[mask,,,]
cifar10$train$y = cifar10$train$y[mask,]

barplot(table(cifar10$train$y),main="Proportion of image types at CIFAR (training data)",
        xlab="Digits")

x_train <- cifar10$train$x
y_train <- cifar10$train$y

x_train <- array_reshape(x_train, c(nrow(x_train), 3072))
x_train <- x_train / 255
y_train <- to_categorical(y_train, 10)


history = NULL
eps = 100

# Prueba con 1 capa
for (unitsHidden in c(64,128,256,512,1024)){
  history.It=NULL
  myfolds = createFolds(y=cifar10$train$y,k=5) 
  k=5
  for(i in 1:k){
    model = keras_model_sequential() 
    model %>% 
      layer_dense(units = unitsHidden, activation = 'relu', input_shape = c(3072)) %>% 
      layer_dense(units = 10) %>% 
      layer_activation('softmax')
    
    summary(model)
    
    model %>% compile(
      loss = 'categorical_crossentropy',
      optimizer = optimizer_rmsprop(),
      metrics = c('accuracy'))
    history.It[[i]] = 
      model %>% fit(
        x_train[-myfolds[[i]],], 
        y_train[-myfolds[[i]],], 
        epochs = eps, 
        batch_size = 128, 
        validation_data = list(x_train[myfolds[[i]],],
                               y_train[myfolds[[i]],]),
        verbose = 2)
  }
  name = paste0("MLP-",unitsHidden)
  history[[name]] = history.It
}


for (unitsHidden in c(256,512,1024)){
  for (unitsHidden2 in c(256,512,1024)){
    history.It=NULL
    myfolds = createFolds(y=cifar10$train$y,k=5) 
    k=5
    for(i in 1:k){
      model = keras_model_sequential() 
      model %>% 
        layer_dense(units = unitsHidden, activation = 'relu', input_shape = c(3072)) %>% 
        layer_dense(units = unitsHidden2, activation = 'relu') %>% 
        layer_dense(units = 10) %>% 
        layer_activation('softmax')
      
      summary(model)
      
      model %>% compile(
        loss = 'categorical_crossentropy',
        optimizer = optimizer_rmsprop(),
        metrics = c('accuracy'))
      history.It[[i]] = 
        model %>% fit(
          x_train[-myfolds[[i]],], 
          y_train[-myfolds[[i]],], 
          epochs = eps, 
          batch_size = 128, 
          validation_data = list(x_train[myfolds[[i]],],
                                 y_train[myfolds[[i]],]),
          verbose = 0)
    }
    name = paste0("Units-",unitsHidden,"-",unitsHidden2)
    history[[name]] = history.It
  }
}


# Añadimos dropout a las tres elegidas
# Primero se lo hacemos a la de 512-1024
for(dout in c(0.3,0.5)){
  history.It=NULL
  myfolds = createFolds(y=cifar10$train$y,k=5) 
  k=5
  for(i in 1:k){
    model = keras_model_sequential() 
    model %>% 
      layer_dense(units = 512, activation = 'relu', input_shape = c(3072)) %>% 
      layer_dropout(rate=dout) %>%
      layer_dense(units = 1024, activation = 'relu') %>% 
      layer_dropout(rate=dout) %>%
      layer_dense(units = 10) %>% 
      layer_activation('softmax')
    
    summary(model)
    
    model %>% compile(
      loss = 'categorical_crossentropy',
      optimizer = optimizer_rmsprop(),
      metrics = c('accuracy'))
    
    history.It[[i]] = 
      model %>% fit(
        x_train[-myfolds[[i]],], 
        y_train[-myfolds[[i]],], 
        epochs = eps, 
        batch_size = 128, 
        validation_data = list(x_train[myfolds[[i]],],
                               y_train[myfolds[[i]],]),
        verbose = 0)
  }
  name = paste0("Units-",unitsHidden,"-",unitsHidden2, "-dropout-",dout)
  history[[name]] = history.It
}


# Y ahora se lo añadimos a las de 1024-512 y 1024-1024
for (unitsHidden in c(1024)){
  for (unitsHidden2 in c(512,1024)){
    for(dout in c(0.3,0.5)){
      history.It=NULL
      myfolds = createFolds(y=cifar10$train$y,k=5) 
      k=5
      for(i in 1:k){
        model = keras_model_sequential() 
        model %>% 
          layer_dense(units = unitsHidden, activation = 'relu', input_shape = c(3072)) %>% 
          layer_dropout(rate=dout) %>%
          layer_dense(units = unitsHidden2, activation = 'relu') %>% 
          layer_dropout(rate=dout) %>%
          layer_dense(units = 10) %>% 
          layer_activation('softmax')
        
        summary(model)
        
        model %>% compile(
          loss = 'categorical_crossentropy',
          optimizer = optimizer_rmsprop(),
          metrics = c('accuracy'))
        
        history.It[[i]] = 
          model %>% fit(
            x_train[-myfolds[[i]],], 
            y_train[-myfolds[[i]],], 
            epochs = eps, 
            batch_size = 128, 
            validation_data = list(x_train[myfolds[[i]],],
                                   y_train[myfolds[[i]],]),
            verbose = 0)
      }
      name = paste0("Units-",unitsHidden,"-",unitsHidden2, "-dropout-",dout)
      history[[name]] = history.It
    }
  }
}


# Dropout de 0.4 para el mejor modelo elegido

history.It=NULL
myfolds = createFolds(y=cifar10$train$y,k=5) 
k=5
for(i in 1:k){
  model = keras_model_sequential() 
  model %>% 
    layer_dense(units = 1024, activation = 'relu', input_shape = c(3072)) %>% 
    layer_dropout(rate=0.4) %>%
    layer_dense(units = 512, activation = 'relu') %>% 
    layer_dropout(rate=0.4) %>%
    layer_dense(units = 10) %>% 
    layer_activation('softmax')
  
  summary(model)
  
  model %>% compile(
    loss = 'categorical_crossentropy',
    optimizer = optimizer_rmsprop(),
    metrics = c('accuracy'))
  
  history.It[[i]] = 
    model %>% fit(
      x_train[-myfolds[[i]],], 
      y_train[-myfolds[[i]],], 
      epochs = eps, 
      batch_size = 128, 
      validation_data = list(x_train[myfolds[[i]],],
                             y_train[myfolds[[i]],]),
      verbose = 0)
}
name = paste0("Units-",unitsHidden,"-",unitsHidden2, "-dropout-",dout)
history[[name]] = history.It


# Tres capas ocultas
for (unitsHidden in c(1024)){
  for (unitsHidden2 in c(512)){
    for (unitsHidden3 in c(128,256,512)){
      history.It=NULL
      myfolds = createFolds(y=cifar10$train$y,k=5) 
      k=5
      for(i in 1:k){
        model = keras_model_sequential() 
        model %>% 
          layer_dense(units = unitsHidden, activation = 'relu', input_shape = c(3072)) %>% 
          layer_dropout(rate=0.3) %>%
          layer_dense(units = unitsHidden2, activation = 'relu') %>% 
          layer_dropout(rate=0.3) %>%
          layer_dense(units = unitsHidden3, activation = 'relu') %>% 
          layer_dropout(rate=0.3) %>%
          layer_dense(units = 10) %>% 
          layer_activation('softmax')
        
        summary(model)
        
        model %>% compile(
          loss = 'categorical_crossentropy',
          optimizer = optimizer_rmsprop(),
          metrics = c('accuracy'))
        history.It[[i]] = 
          model %>% fit(
            x_train[-myfolds[[i]],], 
            y_train[-myfolds[[i]],], 
            epochs = eps, 
            batch_size = 128, 
            validation_data = list(x_train[myfolds[[i]],],
                                   y_train[myfolds[[i]],]),
            verbose = 0)
      }
      name = paste0("Units-",unitsHidden,"-",unitsHidden2, "-", unitsHidden3)
      history[[name]] = history.It
    }
  }
}

# Guardamos las medias en los 5 folds de cada uno de los experimentos
salidasMLP <- c()
mediasMLP <- c()

for(i in 1:(length(history))){
  for (k in 1:5){
    salidasMLP[k] = history[[i]][[k]]$metrics$val_acc[eps]
  }
  name = names(history[i])
  mediasMLP[name] = mean(salidasMLP)
}






