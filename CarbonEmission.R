FLAGS<- flags(
  flag_numeric("nodes1", 32),
  flag_numeric("nodes2", 32),
  flag_numeric("nodes3", 32),
  flag_numeric("batch_size",32),
  flag_string("activation","relu"),
  flag_numeric("learning_rate",0.01),
  flag_numeric("dropout", 0.2)
)

model = keras_model_sequential()

model %>%
  layer_dense(units = FLAGS$nodes1, activation = FLAGS$activation, input_shape = dim(carbonTrainingFinal)[2]) %>%
  layer_dropout(rate=FLAGS$dropout)%>%
  layer_dense(units = FLAGS$nodes2, activation = FLAGS$activation) %>%
  layer_dropout(rate=FLAGS$dropout)%>%
  layer_dense(units = FLAGS$nodes3, activation = FLAGS$activation) %>%
  layer_dropout(rate=FLAGS$dropout)%>%
  layer_dense(units = 1)

model %>% compile(
  loss="mse",
  optimizer=optimizer_adam(lr=FLAGS$learning_rate)
)

model %>%fit(  as.matrix(carbonTrainingFinal),
               carbonTrainingLabels,
               batch_size=FLAGS$batch_size,
               epochs=20,
               validation_data = list(as.matrix(carbonValidationFinal),carbonValidationLabels),verbose = 2
)