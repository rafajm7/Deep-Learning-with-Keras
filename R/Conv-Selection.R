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

img_rows <- 32
img_cols <- 32

x_train <- cifar10$train$x
y_train <- cifar10$train$y

x_train <- array_reshape(x_train, c(nrow(x_train), img_rows,img_cols,3))
input_shape <- c(img_rows,img_cols,3)

x_train <- x_train / 255
y_train <- to_categorical(y_train, 10)

historyConv1=NULL
eps = 60

# Primer tune
for(filter in c(16,32)){
  for(filter2 in c(32,64)){ 
    for(ks1 in c(3,4)){ 
      for(unitsHidden in c(128,256)){ 
        historyConv1.It = NULL
        myfolds = createFolds(y=cifar10$train$y,k=5) 
        k=5
        for(i in 1:k){
          model = keras_model_sequential() 
          model %>%
            layer_conv_2d(filters = filter, kernel_size = c(ks1,ks1), activation = 'relu',
                          input_shape = input_shape) %>% 
            layer_dropout(rate = 0.4) %>% 
            layer_conv_2d(filters = filter2, kernel_size = c(ks1,ks1), activation = 'relu') %>% 
            layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
            layer_dropout(rate = 0.5) %>% 
            
            layer_flatten() %>% 
            layer_dense(units = unitsHidden, activation = 'relu') %>% 
            layer_dropout(rate = 0.5) %>% 
            layer_dense(units = 10, activation = 'softmax')
          
          summary(model)
          
          model %>% compile(
            loss = 'categorical_crossentropy',
            optimizer = optimizer_adadelta(),
            metrics = c('accuracy'))
          
          historyConv1.It[[i]] = 
            model %>% fit(
              x_train[-myfolds[[i]],,,], 
              y_train[-myfolds[[i]],], 
              epochs = eps, 
              batch_size = 128, 
              validation_data = list(x_train[myfolds[[i]],,,],
                                     y_train[myfolds[[i]],]),
              verbose = 0)
        }
        name = paste0("Conv-",filter,"-",filter2,"-KS-",ks1, "-Units-",unitsHidden)
        historyConv1[[name]] = historyConv1.It
      }
    }
  }
}


# Empezamos con 4 filtros, variamos kernel size
for(filter in c(32)){  
  for(filter2 in c(64)){ 
    for(ks1 in c(3,4)){ 
      for(ks2 in c(3,4)){ 
        for(unitsHidden in c(128)){ 
          historyConv2.It = NULL
          myfolds = createFolds(y=cifar10$train$y,k=5) 
          k=5
          for(i in 1:k){
            model = keras_model_sequential() 
            model %>%
              layer_conv_2d(filters = filter, kernel_size = c(ks1,ks1), activation = 'relu',
                            input_shape = input_shape) %>% 
              layer_conv_2d(filters = filter, kernel_size = c(ks1,ks1), activation = 'relu') %>% 
              layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
              layer_dropout(rate = 0.35) %>% 
              
              layer_conv_2d(filters = filter2, kernel_size = c(ks2,ks2), activation = 'relu',
                            input_shape = input_shape) %>% 
              layer_conv_2d(filters = filter2, kernel_size = c(ks2,ks2), activation = 'relu') %>% 
              layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
              layer_dropout(rate = 0.35) %>% 
              
              layer_flatten() %>% 
              layer_dense(units = unitsHidden, activation = 'relu') %>% 
              layer_dropout(rate = 0.5) %>% 
              layer_dense(units = 10, activation = 'softmax')
            
            summary(model)
            
            model %>% compile(
              loss = 'categorical_crossentropy',
              optimizer = optimizer_adadelta(),
              metrics = c('accuracy'))
            historyConv2.It[[i]] = 
              model %>% fit(
                x_train[-myfolds[[i]],,,], 
                y_train[-myfolds[[i]],], 
                epochs = eps, 
                batch_size = 128, 
                validation_data = list(x_train[myfolds[[i]],,,],
                                       y_train[myfolds[[i]],]),
                verbose = 0)
          }
          name = paste0("Conv-",filter,"-",filter,"-",filter2,"-",filter2,"-KS-",ks1,"-",ks2, "-units-",unitsHidden)
          historyConv1[[name]] = historyConv2.It
        }
      }
    }
  }
}

# Variamos filtros: 48-96, y más abajo 64-96 y 64-128

for(filter in c(48)){
  for(filter2 in c(96)){
    for(ks1 in c(3)){
      for(ks2 in c(4)){
        for(unitsHidden in c(128)){
          historyConv1.It = NULL
          myfolds = createFolds(y=cifar10$train$y,k=5) 
          k=5
          for(i in 1:k){
            model = keras_model_sequential()
            model %>%
              layer_conv_2d(filters = filter, kernel_size = c(ks1,ks1), activation = 'relu',
                            input_shape = input_shape) %>%
              layer_conv_2d(filters = filter, kernel_size = c(ks1,ks1), activation = 'relu') %>%
              layer_max_pooling_2d(pool_size = c(2, 2)) %>%
              layer_dropout(rate = 0.4) %>%
              
              layer_conv_2d(filters = filter2, kernel_size = c(ks2,ks2), activation = 'relu') %>%
              layer_dropout(rate = 0.4) %>%
              layer_conv_2d(filters = filter2, kernel_size = c(ks2,ks2), activation = 'relu') %>%
              layer_max_pooling_2d(pool_size = c(2, 2)) %>%
              layer_dropout(rate = 0.4) %>%
              
              layer_flatten() %>%
              layer_dense(units = unitsHidden, activation = 'relu') %>%
              layer_dropout(rate = 0.5) %>%
              layer_dense(units = 10, activation = 'softmax')
            
            summary(model)
            
            model %>% compile(
              loss = 'categorical_crossentropy',
              optimizer = optimizer_adadelta(),
              metrics = c('accuracy'))
            historyConv1.It[[i]] =
              model %>% fit(
                x_train[-myfolds[[i]],,,],
                y_train[-myfolds[[i]],],
                epochs = eps,
                batch_size = 128,
                validation_data = list(x_train[myfolds[[i]],,,],
                                       y_train[myfolds[[i]],]),
                verbose = 0)
          }
          name = paste0("Conv-",filter,"-",filter,"-",filter2,"-",filter2,"-KS-",ks1,"-",ks2, "-units-",unitsHidden)
          historyConv1[[name]] = historyConv1.It
        }
      }
    }
  }
}


for(filter in c(64)){
  for(filter2 in c(96,128)){
    for(ks1 in c(3)){
      for(ks2 in c(4)){
        for(unitsHidden in c(128)){
          historyConv1.It = NULL
          myfolds = createFolds(y=cifar10$train$y,k=5) 
          k=5
          for(i in 1:k){
            model = keras_model_sequential() 
            model %>%
              layer_conv_2d(filters = filter, kernel_size = c(ks1,ks1), activation = 'relu',
                            input_shape = input_shape) %>%
              layer_conv_2d(filters = filter, kernel_size = c(ks1,ks1), activation = 'relu') %>%
              layer_max_pooling_2d(pool_size = c(2, 2)) %>%
              layer_dropout(rate = 0.5) %>%
              
              layer_conv_2d(filters = filter2, kernel_size = c(ks2,ks2), activation = 'relu') %>%
              layer_dropout(rate = 0.5) %>%
              layer_conv_2d(filters = filter2, kernel_size = c(ks2,ks2), activation = 'relu') %>%
              layer_max_pooling_2d(pool_size = c(2, 2)) %>%
              layer_dropout(rate = 0.5) %>%
              
              layer_flatten() %>%
              layer_dense(units = unitsHidden, activation = 'relu') %>%
              layer_dropout(rate = 0.5) %>%
              layer_dense(units = 10, activation = 'softmax')
            
            summary(model)
            
            model %>% compile(
              loss = 'categorical_crossentropy',
              optimizer = optimizer_adadelta(),
              metrics = c('accuracy'))
            
            historyConv1.It[[i]] =
              model %>% fit(
                x_train[-myfolds[[i]],,,],
                y_train[-myfolds[[i]],],
                epochs = eps,
                batch_size = 128,
                validation_data = list(x_train[myfolds[[i]],,,],
                                       y_train[myfolds[[i]],]),
                verbose = 0)
          }
          name = paste0("Conv-",filter,"-",filter,"-",filter2,"-",filter2,"-KS-",ks1,"-",ks2, "-units-",unitsHidden)
          historyConv1[[name]] = historyConv1.It
        }
      }
    }
  }
}


# Obtenemos las medias en los 5 folds de todos los experimentos

salidasConv <- c()
mediasConv <- c()

for(i in 1:(length(historyConv1))){
  for (k in 1:5){
    salidasConv[k] = historyConv1[[i]][[k]]$metrics$val_acc[eps]
  }
  name = names(historyConv1[i])
  mediasConv[name] = mean(salidasConv)
}










