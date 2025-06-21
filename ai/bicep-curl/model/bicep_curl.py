# Data visualization and processing
import numpy as np
import pandas as pd
import os

# TensorFlow/Keras for deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import keras_tuner as kt

# Train-Test split and evaluation metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Important body landmarks for bicep exercise classification
BICEP_LANDMARKS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "LEFT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
]

# Generate column headers for the dataset
DATASET_COLUMNS = ["label"]  # Classification label column

for landmark in BICEP_LANDMARKS:
    DATASET_COLUMNS += [
        f"{landmark.lower()}_x",
        f"{landmark.lower()}_y",
        f"{landmark.lower()}_z",
        f"{landmark.lower()}_v"
    ]


def create_required_directories():
    """Create necessary directories if they don't exist"""
    required_dirs = ["./model", "./keras_tuner_dir", "./keras_tuner_dir/logs"]
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)


def analyze_dataset(dataset_file_path: str):
    """
    Analyze and describe the dataset
    """
    if not os.path.exists(dataset_file_path):
        print(f"Error: Dataset file {dataset_file_path} not found!")
        return None

    dataset = pd.read_csv(dataset_file_path)
    print(f"Dataset columns: {list(dataset.columns.values)}")
    print(f'Dataset shape: {dataset.shape[0]} rows x {dataset.shape[1]} columns\n')
    print(f"Class distribution: \n{dataset['label'].value_counts()}\n")
    print(f"Missing values present: {dataset.isnull().values.any()}\n")

    duplicate_rows = dataset[dataset.duplicated()]
    print(f"Duplicate rows found: {len(duplicate_rows)}")

    return dataset


def clean_duplicate_data(dataset_file_path: str, output_filename: str = "cleaned_train.csv"):
    """
    Remove duplicate rows from dataset and save cleaned version
    """
    if not os.path.exists(dataset_file_path):
        print(f"Error: Dataset file {dataset_file_path} not found!")
        return

    dataset_df = pd.read_csv(dataset_file_path)
    original_size = len(dataset_df)
    dataset_df.drop_duplicates(keep="first", inplace=True)
    cleaned_size = len(dataset_df)

    dataset_df.to_csv(output_filename, sep=',', encoding='utf-8', index=False)
    print(f"Removed {original_size - cleaned_size} duplicate rows")
    print(f"Cleaned dataset saved to {output_filename}")


def round_metric_scores(metric_scores) -> list:
    """Round metric scores to 3 decimal places"""
    return [round(score, 3) for score in metric_scores]


def display_model_architecture(neural_network_model):
    """
    Display the architecture of the neural network model
    """
    print("Neural Network Architecture:")
    print("-" * 40)
    for layer_index, layer in enumerate(neural_network_model.layers):
        layer_units = getattr(layer, 'units', 0)

        if hasattr(layer, "activation"):
            activation_function = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
            print(f"Layer {layer_index + 1}: {layer_units} units, activation: {activation_function}")
        else:
            print(f"Layer {layer_index + 1}: {layer_units} units, activation: None")


def get_optimized_model(hyperparameter_tuner):
    """
    Get the best model found by hyperparameter tuning
    """
    best_hyperparameters = hyperparameter_tuner.get_best_hyperparameters(num_trials=1)[0]
    optimized_model = hyperparameter_tuner.hypermodel.build(best_hyperparameters)

    display_model_architecture(optimized_model)

    print("\nOptimal Hyperparameters:")
    print("-" * 30)
    excluded_params = ["tuner", "activation", "layer", "epoch"]
    for param_name, param_value in best_hyperparameters.values.items():
        if not any(excluded_word in param_name for excluded_word in excluded_params):
            print(f"{param_name}: {param_value}")

    return optimized_model


def build_seven_layer_model(hyperparameters):
    """
    Build a 7-layer deep neural network model for bicep exercise classification
    """
    # Clear any existing models to avoid naming conflicts
    tf.keras.backend.clear_session()

    deep_learning_model = Sequential(name="BicepClassifier_7Layer")

    # Input layer
    deep_learning_model.add(Dense(36, input_dim=36, activation="relu"))

    # Hyperparameters for tuning
    optimal_activation = hyperparameters.Choice('activation_function', values=['relu', 'tanh'])
    hidden_layer_1_units = hyperparameters.Int('hidden_layer_1', min_value=32, max_value=512, step=32)
    hidden_layer_2_units = hyperparameters.Int('hidden_layer_2', min_value=32, max_value=512, step=32)
    hidden_layer_3_units = hyperparameters.Int('hidden_layer_3', min_value=32, max_value=512, step=32)
    hidden_layer_4_units = hyperparameters.Int('hidden_layer_4', min_value=32, max_value=512, step=32)
    hidden_layer_5_units = hyperparameters.Int('hidden_layer_5', min_value=32, max_value=512, step=32)
    optimal_learning_rate = hyperparameters.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # Hidden layers
    deep_learning_model.add(Dense(units=hidden_layer_1_units, activation=optimal_activation))
    deep_learning_model.add(Dense(units=hidden_layer_2_units, activation=optimal_activation))
    deep_learning_model.add(Dense(units=hidden_layer_3_units, activation=optimal_activation))
    deep_learning_model.add(Dense(units=hidden_layer_4_units, activation=optimal_activation))
    deep_learning_model.add(Dense(units=hidden_layer_5_units, activation=optimal_activation))

    # Output layer (binary classification: Correct vs Low bicep curl)
    deep_learning_model.add(Dense(2, activation="softmax"))

    # Compile the model
    deep_learning_model.compile(
        optimizer=Adam(learning_rate=optimal_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return deep_learning_model


def evaluate_model_performance(trained_model, test_features, test_labels_categorical, test_labels_original):
    """
    Evaluate model performance and return metrics
    """
    # Make predictions
    prediction_probabilities = trained_model.predict(test_features, verbose=0)
    predicted_classes = np.argmax(prediction_probabilities, axis=1)
    actual_classes = np.argmax(test_labels_categorical, axis=1)

    # Calculate metrics
    confusion_matrix_result = confusion_matrix(actual_classes, predicted_classes, labels=[0, 1])
    precision_scores, recall_scores, f1_scores, _ = precision_recall_fscore_support(
        actual_classes, predicted_classes, labels=[0, 1], average=None
    )

    return {
        'confusion_matrix': confusion_matrix_result,
        'precision': round_metric_scores(precision_scores),
        'recall': round_metric_scores(recall_scores),
        'f1_score': round_metric_scores(f1_scores)
    }


def main():
    """
    Main training pipeline for bicep exercise classification
    """
    print("=== Bicep Exercise Classification Training ===\n")

    # Create necessary directories
    create_required_directories()

    # Load and analyze training dataset
    print("Loading and analyzing training dataset...")
    training_dataset = analyze_dataset("./train.csv")

    if training_dataset is None:
        print("Cannot proceed without training data. Please ensure train.csv exists.")
        return

    # Encode labels: 'C' (Correct) = 0, 'L' (Low) = 1
    print("Encoding class labels...")
    training_dataset.loc[training_dataset["label"] == "C", "label"] = 0
    training_dataset.loc[training_dataset["label"] == "L", "label"] = 1

    # Handle feature scaling
    scaler_file_path = "./model/input_scaler.pkl"
    if os.path.exists(scaler_file_path):
        with open(scaler_file_path, "rb") as scaler_file:
            feature_scaler = pickle.load(scaler_file)
        print("Loaded existing feature scaler")
    else:
        print("Creating new StandardScaler for feature normalization...")
        feature_scaler = StandardScaler()
        training_features_raw = training_dataset.drop("label", axis=1)
        feature_scaler.fit(training_features_raw)
        with open(scaler_file_path, "wb") as scaler_file:
            pickle.dump(feature_scaler, scaler_file)
        print("Created and saved new feature scaler")

    # Prepare features and labels
    training_features_raw = training_dataset.drop("label", axis=1)
    training_features_scaled = pd.DataFrame(feature_scaler.transform(training_features_raw))
    training_labels = training_dataset["label"]

    # Convert labels to categorical (one-hot encoding)
    training_labels_categorical = to_categorical(training_labels, num_classes=2)

    # Split data for training and validation
    train_features, validation_features, train_labels, validation_labels = train_test_split(
        training_features_scaled.values, training_labels_categorical,
        test_size=0.2, random_state=1234, stratify=training_labels
    )

    print(f"Training set shape: {train_features.shape}")
    print(f"Validation set shape: {validation_features.shape}")

    # Setup training callbacks
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    tensorboard_callback = TensorBoard(
        log_dir="./keras_tuner_dir/logs",
        histogram_freq=1
    )

    # Clear any existing tuner directories to avoid conflicts
    tuner_dir = 'keras_tuner_dir'
    if os.path.exists(tuner_dir):
        import shutil
        shutil.rmtree(tuner_dir)

    # Hyperparameter tuning
    print("\nStarting hyperparameter optimization...")
    hyperparameter_tuner = kt.Hyperband(
        build_seven_layer_model,
        objective='val_accuracy',
        max_epochs=15,
        directory='keras_tuner_dir',
        project_name='bicep_classification_tuning',
        overwrite=True  # This ensures fresh start
    )

    hyperparameter_tuner.search(
        train_features, train_labels,
        validation_data=(validation_features, validation_labels),
        epochs=15,
        callbacks=[early_stopping_callback],
        verbose=1
    )

    # Get the best model
    print("\nRetrieving optimized model...")
    optimized_bicep_classifier = get_optimized_model(hyperparameter_tuner)

    # Train the final model
    print("\nTraining the final optimized model...")
    training_history = optimized_bicep_classifier.fit(
        train_features, train_labels,
        epochs=100,
        batch_size=16,
        validation_data=(validation_features, validation_labels),
        callbacks=[early_stopping_callback, tensorboard_callback],
        verbose=1
    )

    # Evaluate on validation set
    print("\n" + "="*60)
    print("VALIDATION SET EVALUATION")
    print("="*60)

    validation_metrics = evaluate_model_performance(
        optimized_bicep_classifier,
        validation_features,
        validation_labels,
        np.argmax(validation_labels, axis=1)
    )

    print(f"Precision scores: {validation_metrics['precision']}")
    print(f"Recall scores: {validation_metrics['recall']}")
    print(f"F1 scores: {validation_metrics['f1_score']}")
    print(f"Confusion Matrix:\n{validation_metrics['confusion_matrix']}")

    # Test set evaluation (if available)
    if os.path.exists("./test.csv"):
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)

        test_dataset = analyze_dataset("./test.csv")

        if test_dataset is not None:
            # Encode test labels
            test_dataset.loc[test_dataset["label"] == "C", "label"] = 0
            test_dataset.loc[test_dataset["label"] == "L", "label"] = 1

            # Prepare test features
            test_features_raw = test_dataset.drop("label", axis=1)
            test_features_scaled = pd.DataFrame(feature_scaler.transform(test_features_raw))
            test_labels = test_dataset["label"]
            test_labels_categorical = to_categorical(test_labels, num_classes=2)

            # Evaluate on test set
            test_metrics = evaluate_model_performance(
                optimized_bicep_classifier,
                test_features_scaled.values,
                test_labels_categorical,
                test_labels.values
            )

            print(f"Test Precision scores: {test_metrics['precision']}")
            print(f"Test Recall scores: {test_metrics['recall']}")
            print(f"Test F1 scores: {test_metrics['f1_score']}")
            print(f"Test Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    # Save the trained model
    print("\nSaving the trained model...")
    model_save_path = "./model/bicep_classifier_7layer.pkl"
    with open(model_save_path, "wb") as model_file:
        pickle.dump(optimized_bicep_classifier, model_file)

    print(f"Model saved successfully to {model_save_path}")
    print("\n=== Training Pipeline Completed Successfully! ===")


if __name__ == "__main__":
    main()