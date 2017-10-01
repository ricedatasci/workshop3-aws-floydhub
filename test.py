import os
import smartphone6 as sm6
from train import model_func, MODEL_FILE, HIDDEN, BATCH_SIZE

SUBMISSION_FILE = "submission_" + \
    model_func.__name__ + "_" + str(HIDDEN) + ".csv"

if __name__ == "__main__":
    # Load the test data
    xte = sm6.load_test()

    # Crate the model and load the weights
    model = model_func(input_features=xte.shape[1], hidden=HIDDEN)
    model.load_weights(MODEL_FILE)

    # Run the model on the test data
    prob_preds = model.predict(xte, batch_size=BATCH_SIZE)
    sm6.create_submission(prob_preds, submission_fname=SUBMISSION_FILE)
