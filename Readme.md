# CNN served as Rest API for object recognition

Convolutional Neural Network served through a Flask Rest API to identify the objects served as static files. The system uses Swagger to document the API.

## Requirements

Install dependencies in your environment with:
 
```bash
pip install -r requirements.txt
```

You could get more information about how to create a virtual environment in this [link](https://docs.python.org/3/tutorial/venv.html).

## How to use the system

- Activate the virtual environment
- Run in terminal `python app.py`
- Navigate in a browser to the Swagger UI: **http://localhost:5000/swagger**
- If you want to see the images you can navigate to **http://localhost:5000/static/<image_name>** 

## Information about requirements
### Tensorflow
[Tensorflow](https://www.tensorflow.org/): The core open source library to help you develop and train ML models

### Keras
[Keras](https://pypi.org/project/graphviz/) is an API designed for human beings, not machines. Keras follows best practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes the number of user actions required for common use cases, and it provides clear & actionable error messages.

### Flask
[Flask](https://flask.palletsprojects.com/en/1.1.x/) is a micro web framework written in Python. It is classified as a microframework because it does not require particular tools or libraries. 

### Swagger
[Swagger](https://swagger.io/)  is in essence an Interface Description Language for describing RESTful APIs expressed using JSON. Swagger is used together with a set of open-source software tools to design, build, document, and use RESTful web services.


## Probe of Concepts

Under the **poc** folder, there are some scripts with concepts that will be added to the main system in the future. Now, there are examples for transfer learning from a pre-trained model to a custom CNN and samples for Explainable AI (XAI) applied to CNN and visualized using Tensorboard. 