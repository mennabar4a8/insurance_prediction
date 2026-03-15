import gradio as gr
import joblib
import numpy as np

model = joblib.load("insurance_model.pkl")

def predict_insurance(age, sex, bmi, children, smoker, region_northwest, region_southeast, region_southwest):
    data = np.array([[age, sex, bmi, children, smoker,
                      region_northwest, region_southeast, region_southwest]])
    
    prediction = model.predict(data)
    return f"Predicted Cost: {prediction[0]}"

demo = gr.Interface(
    fn=predict_insurance,
    inputs=[
        gr.Number(label="Age"),
        gr.Dropdown([0, 1], label="Sex"),
        gr.Number(label="BMI"),
        gr.Number(label="Children"),
        gr.Dropdown([0, 1], label="Smoker"),
        gr.Dropdown([0, 1], label="Region Northwest"),
        gr.Dropdown([0, 1], label="Region Southeast"),
        gr.Dropdown([0, 1], label="Region Southwest")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Insurance Cost Prediction",
    description="Enter the customer data to predict insurance cost.")
demo.launch()
