import pandas as pd
import re
import keras as kr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization, Concatenate
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.callbacks import Callback


# Load data
data = pd.read_csv("C:/Users/User/Desktop/Σχολη/neuroasafi/news-classification.csv")

# Preprocess text: remove punctuation and numbers
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    return text

# Apply preprocessing to each text in X
X = [preprocess_text(text) for text in data['content']]

# Encode labels for category_level_1
encoder_level_1 = LabelEncoder()
y_level_1 = encoder_level_1.fit_transform(data['category_level_1'])

# Encode labels for category_level_2
encoder_level_2 = LabelEncoder()
y_level_2 = encoder_level_2.fit_transform(data['category_level_2'])

# Split the data into training and validation sets
X_train, X_valid, y_train_level_1, y_valid_level_1, y_train_level_2, y_valid_level_2 = train_test_split(
    X, y_level_1, y_level_2, test_size=0.2, random_state=42
)

unique_words_set = set()
for text in X_train:
    words = text.split()
    unique_words_set.update(words)

# Compute number of unique words
num_unique_words = len(unique_words_set)
print("Number of unique words:", num_unique_words)

# Tokenize text
tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_valid_seq = tokenizer.texts_to_sequences(X_valid)

maxlen = 200  # Max sequence length
X_train_pad = pad_sequences(X_train_seq, padding='post', maxlen=maxlen)
X_valid_pad = pad_sequences(X_valid_seq, padding='post', maxlen=maxlen)


# Define CNN model with multiple output layers
class CustomModel(kr.Model):
    def __init__(self, num_unique_words, output_dim, maxlen, num_classes_level_1, num_classes_level_2):
        super(CustomModel, self).__init__()
        self.embedding_layer = Embedding(input_dim=num_unique_words, output_dim=output_dim, input_length=maxlen)
        self.conv1d_layer = Conv1D(128, 3, activation='relu')
        self.max_pooling_layer = GlobalMaxPooling1D()
        self.dense_layer_1 = Dense(64, activation='relu')
        self.dropout_layer = Dropout(0.5)
        self.dense_output_level_1 = Dense(num_classes_level_1, activation='softmax', name='output_level_1')
        self.dense_output_level_2 = Dense(num_classes_level_2, activation='softmax', name='output_level_2')

    def call(self, inputs):
        x = self.embedding_layer(inputs)
        x = self.conv1d_layer(x)
        x = self.max_pooling_layer(x)
        x = self.dropout_layer(x)
        x = self.dense_layer_1(x)
        output_level_1 = self.dense_output_level_1(x)
        output_level_2 = self.dense_output_level_2(x)
        return {'output_1': output_level_1, 'output_2': output_level_2}



custom_model = CustomModel(num_unique_words, output_dim=100,
                            maxlen=maxlen, num_classes_level_1=len(encoder_level_1.classes_), num_classes_level_2=len(encoder_level_2.classes_))

custom_model.compile(optimizer='adam',
                     loss={'output_1': 'sparse_categorical_crossentropy',
                           'output_2': 'sparse_categorical_crossentropy'},
                     metrics={'output_1': 'accuracy', 'output_2': 'accuracy'})


# Train model
history = custom_model.fit(X_train_pad,
                           {'output_1': y_train_level_1, 'output_2': y_train_level_2},
                           validation_data=(X_valid_pad, {'output_1': y_valid_level_1, 'output_2': y_valid_level_2}),
                           epochs=10, batch_size=64, callbacks=[EarlyStopping(monitor='val_loss', patience=2, min_delta=0.0001)], verbose=2)


max_val_acc1 = max(history.history['val_output_1_accuracy'])
print(f"Highest Validation Accuracy (Level 1): {max_val_acc1}")

max_val_acc2 = max(history.history['val_output_2_accuracy'])
print(f"Highest Validation Accuracy (Level 2): {max_val_acc2}")

# Save the trained model
custom_model.save("C:/Users/User/Desktop/Σχολη/neuroasafi/news_classification_model_multi_level.keras")

# Now, let's use the trained model to categorize content from another CSV file

# Load the new data from CSV
new_data = pd.read_csv("C:/Users/User/Desktop/Σχολη/neuroasafi/new_data.csv")

# Preprocess the text in the new data
new_data['content'] = new_data['content'].apply(preprocess_text)

# Tokenize the preprocessed text using the same tokenizer used during training
X_new_seq = tokenizer.texts_to_sequences(new_data['content'])

# Pad sequences using the same maxlen parameter used during training
X_new_pad = pad_sequences(X_new_seq, padding='post', maxlen=maxlen)

# Load the trained model
custom_model = load_model("C:/Users/User/Desktop/Σχολη/neuroasafi/news_classification_model_multi_level.keras",
                    custom_objects={'CustomModel': CustomModel})

# Make predictions using the trained model
predictions = custom_model.predict(X_new_pad)

predicted_labels_level_1 = encoder_level_1.inverse_transform(predictions['output_1'].argmax(axis=1))
predicted_labels_level_2 = encoder_level_2.inverse_transform(predictions['output_2'].argmax(axis=1))

# Add predicted labels to the new data
new_data['predicted_category_level_1'] = predicted_labels_level_1
new_data['predicted_category_level_2'] = predicted_labels_level_2

# Save the new data with predicted categories
new_data.to_csv("C:/Users/User/Desktop/Σχολη/neuroasafi/new_data_with_predictions_multi_level.csv", index=False)

# Load the data with predictions
predictions_data = pd.read_csv('C:/Users/User/Desktop/Σχολη/neuroasafi/new_data_with_predictions_multi_level.csv')

# Compare predicted categories with actual categories for both levels
correct_predictions_level_1 = (predictions_data['predicted_category_level_1'] == predictions_data['category_level_1']).sum()
correct_predictions_level_2 = (predictions_data['predicted_category_level_2'] == predictions_data['category_level_2']).sum()

total_predictions = len(predictions_data)

# Calculate accuracy for both levels
accuracy_level_1 = correct_predictions_level_1 / total_predictions
accuracy_level_2 = correct_predictions_level_2 / total_predictions

print("Accuracy (Level 1):", accuracy_level_1)
print("Accuracy (Level 2):", accuracy_level_2)