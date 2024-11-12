Here's a **README** file with installation and usage instructions for the spam email classification code:

---

# Spam Email Classification with TensorFlow.js

This project is a simple spam email classification model using a neural network in Node.js with TensorFlow.js. The model is trained to classify an email as **spam** or **not spam** based on word frequencies.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Data Structure](#data-structure)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/dirtaputra/tensorflow-email-spam.git
   cd spam-email-classification
   ```

2. **Install Dependencies**

   Make sure you have [Node.js](https://nodejs.org/) installed. Then, install the required packages.

   ```bash
   npm install
   ```

3. **Prepare Your Data**

   The project includes a sample dataset (`data.js`). Update it as needed to include more data or specific keywords relevant to your classification needs.

## Usage

1. **Run the Model**

   After installing dependencies and preparing the data, run the code to train the model and predict the classification for a sample email.

   ```bash
   node index.js
   ```

2. **Interpreting the Output**

   During training, the code will output the **loss** and **accuracy** for each epoch, indicating how well the model is learning.

   After training, it will output a prediction for a sample email:
   - A value close to `1` indicates **spam**.
   - A value close to `0` indicates **not spam**.

## How It Works

The model uses a neural network with two layers:
- **First Layer**: Processes input features (word frequencies in this case) and applies the ReLU activation function.
- **Second Layer**: Outputs a single value between `0` and `1` using the sigmoid function for binary classification.

The model is compiled with **Adam optimizer** for efficient training and **binary crossentropy loss** for measuring classification error.

### Model Training

The model is trained for 100 epochs using small batches of data to improve generalization.

### Data Prediction

A sample email represented as word frequencies (e.g., `promo`, `discount`, `free`) is fed into the model to predict if it’s likely **spam** or **not spam**.

## Data Structure

The data is stored in `data.js`, formatted as two arrays:
- **data**: An array of feature arrays, where each sub-array represents word counts (e.g., `[promoCount, discountCount, freeCount]`).
- **labels**: An array of `0` or `1` values, where `1` represents **spam** and `0` represents **not spam**.

Example structure:

```javascript
module.exports = {
  data: [
    [5, 0, 1], // Frequencies for "promo", "discount", and "free"
    [0, 3, 0],
    [2, 2, 1],
    // Add more data points as needed
  ],
  labels: [1, 0, 1] // 1 = Spam, 0 = Not Spam
};
```

---

This project provides a basic framework for spam classification and can be expanded with more data and features for better accuracy.