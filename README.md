# Fully Homomorphic Encryption (FHE) Simple Model Deployment

This project demonstrates a *basic* example of using the `concrete-ml` library to train a logistic regression model, deploy it using a Flask server, and perform encrypted predictions using Fully Homomorphic Encryption (FHE).

**IMPORTANT:** This project is intended for *educational purposes only* and is **not** production-ready.  It uses a simplified setup and a Flask development server, not suitable for production environments.

## Project Structure

- `app.py`: Implements the API that receives encrypted data, runs the model, and returns the encrypted results.
- `client.py`: Client script that sends encrypted data to the server and receives the encrypted response.
- `train.py`: Script to train and compile the logistic regression model.
- `test_no_flask.py`: Tests the deployment module of `concrete.ml` without using the REST API.

## Requirements

- Python 3.8+
- Libraries: `concrete-ml==1.8.0`, `numpy`, `requests`, `flask`

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/msiciliag/concreteML_test
    cd concreteML_test
    ```

2. Install the dependencies:
    ```sh
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage

### Train the Model

Run the `train.py` script to train on random data and compile the logistic regression model.

To clean up the model files, run the following command:

```sh
rm -rf /tmp/fhe_client_server_files/
```

### Start the Flask Server

Run the `app.py` script to start the Flask server (runs on port 5001).


### Send Request with Encrypted Data

Run the `client.py` script to send encrypted data to the Flask server and receive the encrypted response.


### Check Results without Using the Server

To check the results without using Flask, you can run the `test_no_flask.py` script to test the deployment module of `concrete.ml` without using REST API requests.

**_NOTE:_**  `test_no_flask.py` and `client.py` should give the same results as they are using the same data and model.

### References
For more details on how to use concrete.ml for model deployment with FHE, refer to the [concrete.ml documentation](https://github.com/zama-ai/concrete-ml/blob/main/docs/guides/client_server.md).

