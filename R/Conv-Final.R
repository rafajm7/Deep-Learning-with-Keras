library(tensorflow)
library(keras)

set.seed(12345)
cifar10 <- dataset_cifar10()

img_rows <- 32
img_cols <- 32

x_train <- cifar10$train$x
y_train <- cifar10$train$y
x_test <- cifar10$test$x
y_test <- cifar10$test$y

x_train <- array_reshape(x_train, c(nrow(x_train), img_rows,img_cols,3))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows,img_cols,3))
input_shape <- c(img_rows,img_cols,3)

x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

filter = 64
filter2 = 96
ks1 = 3
ks2 = 4
unitsHidden = 128


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

historyConv = model %>% fit(
              x_train,
              y_train,
              epochs = 60,
              batch_size = 128,
              verbose = 2)

str(historyConv)

score <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)
