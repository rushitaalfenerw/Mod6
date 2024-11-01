# Install required packages if not already installed
if (!require(tensorflow)) install.packages("tensorflow")
if (!require(keras)) install.packages("keras")

library(tensorflow)
library(keras)

# Load the Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
train_data <- fashion_mnist$train
test_data <- fashion_mnist$test

# Extract images and labels
train_images <- train_data$x / 255
train_labels <- train_data$y
test_images <- test_data$x / 255
test_labels <- test_data$y

# Reshape data
train_images <- array_reshape(train_images, dim = c(dim(train_images)[1], 28, 28, 1))
test_images <- array_reshape(test_images, dim = c(dim(test_images)[1], 28, 28, 1))

# Define the CNN model
model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(28, 28, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

# Compile the model
model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)

# Train the model
history <- model %>% fit(
  train_images, train_labels, 
  epochs = 10, 
  validation_data = list(test_images, test_labels)
)

# Make predictions
predictions <- model %>% predict(test_images)
predicted_classes <- apply(predictions, 1, which.max) - 1

# Print predictions for the first two images
class_names <- c("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
for (i in 1:2) {
  cat("Image", i, "True Label:", class_names[test_labels[i] + 1], 
      "Predicted Label:", class_names[predicted_classes[i] + 1], "\n")
}
