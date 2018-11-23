# paper abstract classification for 6 journals
#       Ecography
#       Journal of biogeography
#       Global ecology and biogeography
#       Diversity and distributions
#       Ecology
#       Molecular ecology
## ---- upload abstracts ----
root <- "~/Desktop/A/Me/Papers_abstracts"
# file connection
# abstract parser reads the text files and create the numeric labels 
abstract_parser <- function(dir_path){
        all_journal <- list()
        journals_dir <- dir(dir_path)
        #attr(journals_dir, "names") <- dir(dir_path)
        label_count <- 1
        for (j in 1:length(journals_dir)){
                file_dir <- file.path(dir_path, journals_dir[j])
                file_name <- dir(file_dir, pattern="*abstract.txt")
                all_journal[[(j*2)-1]] <- readLines(file.path(file_dir, file_name), encoding = "UTF-8")
                # data <- strsplit(file_name, split="_")[[1]][1]
                # assign(data,readLines(file.path(file_dir, file_name), encoding = "UTF-8"))
                # all_journal[[(j*2)-1]] <- get(data)
                all_journal[[j*2]] <- rep(j, length(all_journal[[(j*2)-1]]))
        }
        all_journal
}
all_journal <- abstract_parser(root)
## ---- data segmentation ----
# 25% of the data is for testing
# perc_test: how much of the total data is allocated for testing
# perc_val: how much of the training data is allocated for validation
# name_segment: test, val, train
# seed: value to define set.seed()
segment_data <- function(abstract_list, perc_test, perc_val, seed){
        set.seed(seed)
        test <- list()
        train <- list()
        val <- list()
        all_data <- list()
        for(j in 1:(length(abstract_list)/2)){
                tmp_data <- abstract_list[[(j*2)-1]]
                tmp_label <- abstract_list[[(j*2)]]
                tmp_aux <- 1:length(tmp_data)
                tmp_sample_test <- sample(tmp_aux, round(length(tmp_data)*perc_test))
                test[[(j*2)-1]] <- as.vector(tmp_data[tmp_sample_test])
                test[[(j*2)]] <- tmp_label[tmp_sample_test]
                tmp_sample <- tmp_aux[-tmp_sample_test]
                tmp_sample_val <- sample(tmp_sample, length(tmp_sample)*perc_val)
                val[[(j*2)-1]] <- tmp_data[tmp_sample_val]
                val[[(j*2)]] <- tmp_label[tmp_sample_val]
                train[[(j*2)-1]] <- tmp_data[-c(tmp_sample_val,tmp_sample_test)]
                train[[(j*2)]] <- tmp_label[-c(tmp_sample_val,tmp_sample_test)]
        }
        all_data[[1]] <- train
        all_data[[2]] <- val
        all_data[[3]] <- test
        all_data
}
all_segmented <- segment_data(all_journal, .25, .25, 55)
extract_data_label <- function(data, type){
        if(type == "data"){
                tmp <- data[seq(1,(length(data))-1,2)]
        }
        if(type == "label"){
                tmp <- data[seq(2,length(data),2)]
        }
        out <- unlist(tmp)
}
train <- all_segmented[[1]]
train_data <- extract_data_label(train, "data")
train_label <- extract_data_label(train, "label")
val  <- all_segmented[[2]]
val_data <- extract_data_label(val, "data")
val_label <- extract_data_label(val, "label")
test <- all_segmented[[3]]
test_data <- extract_data_label(test, "data")
test_label <- extract_data_label(test, "label")
## ---- tokenization ----
maxlen <- 300
max_words <- 10000

train_val <- c(train_data, val_data)
tokenizer <- text_tokenizer(num_words = max_words) %>% 
        fit_text_tokenizer(train_val)
word_index <- tokenizer$word_index
# training sequences
train_sequences <- texts_to_sequences(tokenizer, train_data)
train_x <- pad_sequences(train_sequences, maxlen =  maxlen)
train_y <-  to_categorical(train_label)
# validation sequences
val_sequences <- texts_to_sequences(tokenizer, val_data)
val_x <- pad_sequences(val_sequences, maxlen)
val_y <- to_categorical(val_label)
## ---- embedding with rnn ----
# model
model <- keras_model_sequential() %>% 
        layer_embedding(input_dim = 10000, output_dim = 32, input_length = maxlen) %>% 
        layer_simple_rnn(units=32, activation = "relu") %>% 
        layer_dense(units=7, activation="softmax")
# compiler
model %>% compile(
        loss= "categorical_crossentropy",
        metrics = c("accuracy"),
        optimizer = "rmsprop"
)
# fit
history <- model %>% fit(
        train_x, 
        train_y, 
        epochs = 10, 
        batch_size = 32, 
        validation_data = list(val_x, val_y)
)
## ---- embedding with gru ----
# model
model_gru <- keras_model_sequential() %>% 
        layer_embedding(input_dim = 10000, output_dim = 32, input_length = maxlen) %>% 
        layer_gru(units=32, activation = "relu", return_sequences = T) %>%
        layer_gru(units=64, activation = "relu", dropout = .5, recurrent_dropout = .3, 
                  return_sequences = T) %>% 
        layer_gru(units=128, activation = "relu", dropout = .2, recurrent_dropout = .1) %>% 
        layer_dense(units=7, activation="softmax")
# compiler
model_gru %>% compile(
        loss= "categorical_crossentropy",
        metrics = c("accuracy"),
        optimizer = "rmsprop"
)
# fit
history_gru <- model_gru %>% fit(
        train_x, 
        train_y, 
        epochs = 10, 
        batch_size = 32, 
        validation_data = list(val_x, val_y)
)
plot(history_gru)
## ---- using GloVe ----
# using the pretrained embedding 
glove_dir <- "~/Downloads/glove.6B/"
lines <- readLines(file.path(glove_dir, "glove.6B.100d.txt"))
embedding_index <- new.env(hash=T, parent=emptyenv())
for(i in 1:length(lines)){
        line <- lines[[i]]
        values <- strsplit(line, " ")[[1]]
        word <- values[[1]]
        # as double simple trasform the character values to numbers
        embedding_index[[word]] <- as.double(values[-1])
}
embedding_dim <- 100
embedding_matrix <- array(0, c(max_words, embedding_dim))
# assign an embedding vector to each of the words in the index
for(word in names(word_index)){
        index <- word_index[[word]]
        if (index < max_words){
                embedding_vector <- embedding_index[[word]]
                if(!is.null(embedding_vector)){
                        embedding_matrix[index+1,] <- embedding_vector
                }
        }
}
# define the model
model_glove <- keras_model_sequential() %>% 
        layer_embedding(input_dim = 10000, output_dim = embedding_dim,
                        input_length = maxlen) %>% 
        layer_gru(units=32, activation = "relu", return_sequences = T) %>%
        layer_gru(units=64, activation = "relu", dropout = .5, recurrent_dropout = .3) %>%
        layer_dense(units=7, activation="softmax")
# get the embedding layer and change the weigths for Glove embeddings
get_layer(model_glove, index = 0) %>% 
        set_weights(list(embedding_matrix)) %>% 
        freeze_weights()
# compilation scheme
model_glove %>% compile(
        loss = "categorical_crossentropy",
        metrics = c("accuracy"),
        optimizer = "rmsprop"
)
## ---- deeper gru ---- 
# model
model_gru_longer <- keras_model_sequential() %>% 
        layer_embedding(input_dim = 10000, output_dim = 32, input_length = maxlen) %>% 
        layer_gru(units=32, activation = "relu", return_sequences = T) %>%
        layer_gru(units=64, activation = "relu", dropout = .5, recurrent_dropout = .3, 
                  return_sequences = T) %>% 
        layer_gru(units=128, activation = "relu", dropout = .2, recurrent_dropout = .1,
                  return_sequences = T) %>% 
        layer_gru(units=524, activation = "relu") %>% 
        layer_dense(units=7, activation="softmax")
# compiler
model_gru_longer %>% compile(
        loss= "categorical_crossentropy",
        metrics = c("accuracy"),
        optimizer = "rmsprop"
)
# fit
history_gru_longer <- model_gru_longer %>% fit(
        train_x, 
        train_y, 
        epochs = 20, 
        batch_size = 32, 
        validation_data = list(val_x, val_y)
)
history_glove <- model_glove %>% fit(
        train_x, 
        train_y, 
        epochs = 10, 
        batch_size = 32, 
        validation_data = list(val_x, val_y)
)
## ---- 1d CNN ----

## ---- ResNet ----
