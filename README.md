# SignNet-A-Deep-Learning-Approach-to-Sign-Language-Recognition

## Project Overview

This project, titled **BSLNet**, aims to create a machine learning model that recognizes and interprets British Sign Language (BSL) fingerspelling using Artificial Neural Networks (ANN). The primary objective was to develop an algorithm capable of recognizing BSL alphabet fingerspelling through a webcam-based graphical user interface (GUI). This initiative addresses the challenge of language deprivation among children with hearing impairments, enabling them to engage in basic communication through sign language recognition.

## Features

- **Data Collection**: Images of BSL fingerspelling were captured using OpenCV and a webcam. Data was collected methodically, with multiple snapshots of each letter taken at different angles and scales to improve model generalization.
- **Model Architecture**: A Sequential neural network was employed, featuring an input layer, hidden layers with ReLU activation functions, and a softmax output layer for multi-class classification.
- **Training and Validation**: The model was trained on 80% of the dataset, with the remaining 20% used for validation. The Adam optimizer and categorical cross-entropy loss function were used to train the model.
- **Real-time Prediction**: The model was tested in real-time, predicting BSL letters from webcam feed by analyzing hand landmarks, proving its effectiveness in sign language recognition.
- **User Interface**: A simple GUI was developed, allowing users to interact with the model and view real-time predictions.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/BSLNet.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd BSLNet
   ```
3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset consists of images representing BSL alphabet fingerspelling, captured using a webcam. The images were processed using the MediaPipe library to identify hand landmarks, which were used as input features for the model.

## Model Training

The core model was designed using a Sequential neural network architecture:

- **Input Layer**: Accepts flattened landmark coordinates.
- **Hidden Layers**: Includes layers with 256 and 128 units, each followed by a Dropout layer to prevent overfitting.
- **Output Layer**: Uses a softmax layer for classifying BSL letters.

### Training Steps:

1. **Data Preprocessing**: Images were resized, normalized, and hand landmarks were identified.
2. **Model Compilation**: The model was compiled using the Adam optimizer and categorical cross-entropy loss function.
3. **Training**: The model was trained on 80% of the dataset, with 20% used for validation.
4. **Evaluation**: The model’s performance was evaluated using accuracy and loss metrics, with results visualized using matplotlib.

## Usage

To use the trained model for real-time BSL recognition:

1. **Run the prediction script**:
   ```bash
   python predict.py
   ```
2. **Provide input**: Use a webcam to perform BSL fingerspelling.
3. **View Results**: The recognized letters will be displayed in real-time on the screen.

## Results

The model achieved strong performance, successfully recognizing most BSL letters. Real-time testing demonstrated the model’s practical application in enabling communication through sign language, although some similar letters posed challenges.

## Contributions

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **TensorFlow and Keras**: For providing powerful tools to build and train neural networks.
- **OpenCV**: For image processing and real-time video analysis.
- **The BSL Dataset Community**: For contributing resources that made this project possible.
