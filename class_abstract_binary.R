# tokenize all the available abstracts from:
#       Journal of biogeography
#       Diversity and distributions
# the first NN will be a binary classification using divdis and jbio
## ---- get.abstracts ----
root <- "~/Desktop/A/Me/Papers_abstracts"
journals_dir <- c("diversity_distributions", "Jbiogeography")
# file connection
file_divdis <- file.path(root, journals_dir[1], "divdis_abstract.txt")
file_jbio <- file.path(root, journals_dir[2], "jbioge_abstract.txt")
# read the lines in the connection
text_divdis <- readLines(file_divdis)
n_divdis <- length(text_divdis)
text_jbio <- readLines(file_jbio)
n_jbio <- length(text_jbio)
labels_divdis <- rep(0, n_divdis)
labels_jbio <- rep(1, n_jbio )

## ---- data.segmentation ----
# segmentation of the data into training, val and test
# 25% of the data for testing
set.seed(5)
sample_div_dis <- sample(1:n_divdis, round(n_divdis*.25))
sample_jbio <- sample(1:n_jbio , round(n_jbio *.25))
# test data and labels
data_test <- c(text_divdis[sample_div_dis], text_jbio[sample_jbio])
label_test <- c(labels_divdis[sample_div_dis], labels_jbio[sample_jbio])
# train data and labels
data_train <- c(text_divdis[-sample_div_dis], text_jbio[-sample_jbio])
label_train <- c(labels_divdis[-sample_div_dis], labels_jbio[-sample_jbio])

## ---- tokenization ---- 
# tokenizer
maxlen <- 100
training_samples <- 1800
validation_samples <- 1163
max_words <- 10000

tokenizer <- text_tokenizer(num_words = max_words) %>%
        fit_text_tokenizer(data_train)
sequences <- texts_to_sequences(tokenizer, data_train)
# pad the sequences
data <- pad_sequences(sequences, maxlen = maxlen)
# convert the vector of labels into an array
labels <- label_train
#labels <- to_categorical(labels)
# divide the dat into val and train
indices <- sample(1:nrow(data))
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples+1):(training_samples+validation_samples)]

x_train <- data[training_indices,]
#dim(x_train)
x_val <- data[validation_indices,]
#dim(x_val)

y_train <- labels[training_indices]
#dim(y_train)
#y_train <- to_categorical(y_train)
y_val <- labels[validation_indices]
#y_val <- to_categorical(y_val)

## ---- NN.define.model ----
model <- keras_model_sequential() %>% 
        layer_embedding(input_dim = 10000, output_dim = 8, input_length = maxlen) %>% 
        layer_flatten() %>% 
        layer_dense(units=1, activation = "sigmoid")

## ---- NN.compile.model ----
model %>%  compile(
        optimizer = "rmsprop",
        metrics = c("accuracy"),
        loss = "binary_crossentropy"
)

## ---- plot.validation ----
history <- model %>% fit(
        x_train, 
        y_train, 
        epochs = 10, 
        batch_size = 32, 
        validation_data = list(x_val, y_val)
)
plot(history)

## ---- adding.dense.layer ----
model <- keras_model_sequential() %>% 
        layer_embedding(input_dim = 10000, output_dim = 8, input_length = maxlen) %>% 
        layer_flatten() %>% 
        layer_dense(units = 32, activation="relu") %>% 
        layer_dense(units=1, activation = "sigmoid")
# compile
model %>%  compile(
        optimizer = "rmsprop",
        metrics = c("accuracy"),
        loss = "binary_crossentropy"
)
history <- model %>% fit(
        x_train, 
        y_train, 
        epochs = 10, 
        batch_size = 32, 
        validation_data = list(x_val, y_val)
)
plot(history)

## ---- a simple rnn ----
model_rnn <- keras_model_sequential() %>% 
        layer_embedding(input_dim = 10000, output_dim = 8, input_length = maxlen) %>% 
        layer_simple_rnn(units = 32) %>% 
        layer_dense(units=1, activation = "sigmoid")
# compile
model_rnn %>%  compile(
        optimizer = "rmsprop",
        metrics = c("accuracy"),
        loss = "binary_crossentropy"
)

history <- model_rnn %>% fit(
        x_train, 
        y_train, 
        epochs = 10, 
        batch_size = 32, 
        validation_data = list(x_val, y_val)
)
# the simple rnn is not really poweful and does not perform better than a dense layer
## ---- lstm layer ----
model_lstm <- keras_model_sequential() %>% 
        layer_embedding(input_dim = 10000, output_dim = 8, input_length = maxlen) %>% 
        layer_lstm(units = 32) %>% 
        layer_dense(units=1, activation = "sigmoid")
# compile
model_lstm %>%  compile(
        optimizer = "rmsprop",
        metrics = c("accuracy"),
        loss = "binary_crossentropy"
)

history_lstm <- model_lstm %>% fit(
        x_train, 
        y_train, 
        epochs = 10, 
        batch_size = 32, 
        validation_data = list(x_val, y_val)
)
model_lstm %>%  save_model_hdf5("model_lstm_paper_abstract.h5")
## ---- recurrent dropout (gru) ----
model_gru <- keras_model_sequential() %>% 
        layer_embedding(input_dim = 10000, output_dim = 8, input_length = maxlen) %>% 
        layer_gru(units = 32, recurrent_dropout = .2, dropout = .2) %>% 
        layer_dense(units=1, activation = "sigmoid")
# compile
model_gru %>%  compile(
        optimizer = "rmsprop",
        metrics = c("accuracy"),
        loss = "binary_crossentropy"
)

history <- model_gru %>% fit(
        x_train, 
        y_train, 
        epochs = 10, 
        batch_size = 32, 
        validation_data = list(x_val, y_val)
)
# seems like overfitting is not a big problem in this analysis. 
## ---- increase sequence length ----
# here the maxlen was increased to 200
maxlen <- 200
training_samples <- 1800
validation_samples <- 1163
max_words <- 10000

tokenizer <- text_tokenizer(num_words = max_words) %>%
        fit_text_tokenizer(data_train)
sequences <- texts_to_sequences(tokenizer, data_train)
# pad the sequences
data <- pad_sequences(sequences, maxlen = maxlen)
# convert the vector of labels into an array
labels <- label_train
#labels <- to_categorical(labels)
# divide the dat into val and train
indices <- sample(1:nrow(data))
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples+1):(training_samples+validation_samples)]

x_train <- data[training_indices,]
dim(x_train)
x_val <- data[validation_indices,]
dim(x_val)

y_train <- labels[training_indices]
y_val <- labels[validation_indices]


## ---- longer.sequence.lenght  ----
model_lstm_200 <- keras_model_sequential() %>% 
        layer_embedding(input_dim = 10000, output_dim = 8, input_length = maxlen) %>% 
        layer_flatten() %>% 
        layer_dense(units = 32, activation="relu") %>% 
        layer_dense(units=1, activation = "sigmoid")
# compile
model_lstm_200 %>%  compile(
        optimizer = "rmsprop",
        metrics = c("accuracy"),
        loss = "binary_crossentropy"
)

history <- model_lstm_200 %>% fit(
        x_train, 
        y_train, 
        epochs = 10, 
        batch_size = 32, 
        validation_data = list(x_val, y_val)
)
plot(history)

## ----  lstm_model_200 ----
model_lstm_200 <- keras_model_sequential() %>% 
        layer_embedding(input_dim = 10000, output_dim = 8, input_length = maxlen) %>% 
        layer_lstm(units = 32) %>% 
        layer_dense(units=1, activation = "sigmoid")
# compile
model_lstm_200 %>%  compile(
        optimizer = "rmsprop",
        metrics = c("accuracy"),
        loss = "binary_crossentropy"
)

history <- model_lstm_200 %>% fit(
        x_train, 
        y_train, 
        epochs = 10, 
        batch_size = 32, 
        validation_data = list(x_val, y_val)
)
## ---- stacking rnn ----
model_gru_200_stck <- keras_model_sequential() %>%
        layer_embedding(input_dim = 10000, output_dim = 8, input_length = maxlen) %>% 
        layer_gru(units = 32, return_sequences = T) %>%
        layer_gru(units = 64, activation = "relu") %>% 
        layer_dense(units=1, activation = "sigmoid")
# compile
model_gru_200_stck %>%  compile(
        optimizer = "rmsprop",
        metrics = c("accuracy"),
        loss = "binary_crossentropy"
)

history_gru_200_stck <- model_gru_200_stck %>% fit(
        x_train, 
        y_train, 
        epochs = 10, 
        batch_size = 32, 
        validation_data = list(x_val, y_val)
)
## --- evaluate model lstm ----
# evaluate the model using a lstm rnn
# tokenize the test data
sequences_test <- texts_to_sequences(tokenizer, data_test)
x_test <- pad_sequences(sequences_test, maxlen)
y_test <- label_test
model_lstm %>% evaluate(x_test, y_test)
