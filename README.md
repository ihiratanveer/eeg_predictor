# EEG Predictor Flask Application

Welcome to the EEG Predictor Flask Application! This application utilizes a machine learning model to classify EEG (Electroencephalogram) files as abnormal (1) or normal (0). It provides a user-friendly interface to upload an EDF (European Data Format) file and get the prediction result.

## Features

Upload an EDF file for classification
Predict if the EEG file is abnormal (1) or normal (0)
User-friendly web interface
## Installation

To run the EEG Predictor Flask Application locally, please follow the instructions below:

Clone this repository to your local machine or download the source code as a ZIP file.
Ensure that you have Python 3.x installed on your system.
Open a terminal or command prompt and navigate to the project's directory.
## Usage

Install the required dependencies by running the following command:
```shell
pip install flask
```

Start the Flask application by running the following command:
```shell
python app.py
```

Open a web browser and go to http://localhost:5000 to access the application.
Click on the "Choose File" button to select an EDF file from your local machine.
After selecting the file, click the "Predict" button to initiate the classification process.
Wait for the prediction result to be displayed on the screen.
You can repeat the process with different EDF files as needed.
**Note:** The machine learning model used for classification is located at the backend. You do not need to interact with the model directly; the application handles it for you.



