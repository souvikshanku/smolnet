"""Module for running the app.
"""
import pickle

import gradio as gr
import numpy as np


with open('my_smolnet.pkl', 'rb') as f:
    smolnet = pickle.load(f)

digits = list(range(10))

def predict(image):
    """Predict what digit is in the image.
    """
    x = np.reshape(image, (784, 1)) / 255
    probs = smolnet.predict(x.reshape(784,1))
    confidences = {str(digits[i]): probs[i][0] for i in range(10)}

    return confidences

title = "Welcome to smolnet MNIST classifier!"

head = (
  "<center>"
  "Draw your single digit number in the canvas."
  "</center>"
)


sp =  gr.Sketchpad(shape=(28, 28), brush_radius=1)

gr.Interface(
    fn=predict,
    inputs=sp,
    outputs="label",
    title=title,
    description=head,
    live=False
).launch()
