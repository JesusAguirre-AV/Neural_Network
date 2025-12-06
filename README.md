#Project 3: Neural Networks

## Instructions on how to run

## File Manifest
### main.py
Coordinates the full workflow of the project.
- Calls the database functions to extract their features.
- Splits th data between training and testing sets.
- Runs multiple models and compares them for accuracy.
- Generates Kaggle submission.

### database.py
- Handles aspects of dataset creation and management.
- Loads, resamples, and normalizes raw audio files to a consistent sampling rate.
- Extracts features and aggregates them into a unified dataset.
- Outputs tain_features and test_features as csv files to allow them to be reused.

### ConvolutionalNeuralNetworks.py
- CNN architecture designed to handle spectrogram classification.
- Uses a filtration system and a kernel to allow convolution.
- Also includes drop out rate to help prevent overfitting.

### MultilayerPerceptron.py
- MlP architecture designed to handle spectrogram classification.
- Allows for hyperparameter fine tuning.
- Also includes drop out rate to help prevent overfitting.

### TransferLearning.py
- Uses ResNet model as a pretrained model with spectrogram modality.
- Allows for fine-tuning via allowing to freeze certain layers.

## Contributors
- Ben | Multilayer Perceptron
- Greg | Convolution Neural Networks
- Carly Salazar | Sweeps and testing
- Jesus Aguirre | Transfer Learning

## Performance