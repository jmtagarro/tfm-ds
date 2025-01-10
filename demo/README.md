
# Web Application Demo

This demo allows you to run three different models based on the modalities used: Text, Image, and Text & Image combined. The application consists of:

1. **Three Flask-based models web services, one per model**:
   - `dmrl-text` running on port `8081`.
   - `dmrl-image` running on port `8082`.
   - `dmrl-multi` running on port `8083`.

2. **A Node.js web server**:
   - Hosts the web application at `http://localhost:8080`.

## Prerequisites

### System Requirements
- **Python 3.8+**: To run the Flask models.
- **Cornac 3.30**: To run the models.
- **Node.js and npm**: To run the Node.js server.

### Install Required Dependencies
1. **Python Libraries**:
   Install Flask and other required Python libraries:
   ```bash
   pip3 install Flask
   ```

2. **Node.js and npm**:
   Install Node.js and npm if not already installed. On Ubuntu, use:
   ```bash
   sudo apt update
   sudo apt install -y nodejs npm
   ```
   Verify installation:
   ```bash
   node -v
   npm -v
   ```

## Setting Up the Demo

### Step 1: Launch the Flask Models
1. Navigate to the directories where the Flask models (`dmrl-text`, `dmrl-image`, and `dmrl-multi`) are located.
2. Start each model with Flask, specifying the appropriate port. For example:
   - **dmrl-text**:
     ```bash
     FLASK_APP='cornac.serving.app' \ 
     MODEL_PATH='models/dmrl-text' \
     MODEL_CLASS='cornac.models.BPR' \
     flask run --host=127.0.0.1 --port=8081
     ```
   - **dmrl-image**:
     ```bash
     FLASK_APP='cornac.serving.app' \ 
     MODEL_PATH='models/dmrl-image' \
     MODEL_CLASS='cornac.models.BPR' \
     flask run --host=127.0.0.1 --port=8082
     ```
   - **dmrl-multi**:
     ```bash
     FLASK_APP='cornac.serving.app' \ 
     MODEL_PATH='models/dmrl-multi' \
     MODEL_CLASS='cornac.models.BPR' \
     flask run --host=127.0.0.1 --port=8083
     ```

   Each Flask instance should display a message similar to:
   ```
   * Running on http://127.0.0.1:808X
   ```

### Step 2: Set Up and Run the Node.js Web Application
1. Navigate to the `demo` folder:
   ```bash
   cd demo
   ```

2. Install the Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the Node.js server:
   ```bash
   node app.js
   ```

   The server will start and listen on port `8080`.

### Step 3: Open the Web Application
1. Open your browser and navigate to:
   ```
   http://localhost:8080
   ```

2. Use the buttons and drop-down menu to explore recommendations based on different modalities:
   - **Text Modality Only** (Port `8081`)
   - **Image Modality Only** (Port `8082`)
   - **Text & Image Modalities** (Port `8083`)

## Troubleshooting

- **Flask Error: `ModuleNotFoundError`**:
  Ensure the Flask apps are correctly configured, and all necessary libraries are installed.

- **Node.js Error: `Module Not Found`**:
  Ensure you ran `npm install` in the `demo` folder before starting the server.

- **Connection Error**:
  Ensure all three Flask models are running and accessible on their respective ports (`8081`, `8082`, `8083`).

## Directory Structure

```
demo/
├── app.js                 # Node.js server script
├── public/
│   ├── index.html         # Frontend HTML file
│   └── ...                # Other frontend assets
├── package.json           # Node.js dependencies
├── README.md              # This README file
models/
├── dmrl-text              # Text modality model
├── dmrl-image             # Image modality model
├── dmrl-multi             # Text and image modalities model
```

## License
This project is licensed under the MIT License.
