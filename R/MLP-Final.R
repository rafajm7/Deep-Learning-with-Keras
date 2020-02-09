library(tensorflow)
library(keras)

set.seed(12345)

cifar10 <- dataset_cifar10()

x_train <- cifar10$train$x
y_train <- cifar10$train$y
x_test <- cifar10$test$x
y_test <- cifar10$test$y
x_train <- array_reshape(x_train, c(nrow(x_train), 3072))
x_test <- array_reshape(x_test, c(nrow(x_test), 3072))


x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

unitsHidden = 1024
unitsHidden2 = 512
unitsHidden3 = 128
dout = 0.3


model = keras_model_sequential() 
model %>% 
  layer_dense(units = unitsHidden, activation = 'relu', input_shape = c(3072)) %>% 
  layer_dense(units = unitsHidden2, activation = 'relu') %>% 
  layer_dropout(rate=dout) %>%
  layer_dense(units = unitsHidden3, activation = 'relu') %>% 
  layer_dropout(rate=dout) %>%
  layer_dense(units = 10) %>% 
  layer_activation('softmax')

summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy'))

history = model %>% fit(
          x_train, 
          y_train, 
          epochs = 100, 
          batch_size = 128, 
          verbose = 2)

str(history)

score <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)


