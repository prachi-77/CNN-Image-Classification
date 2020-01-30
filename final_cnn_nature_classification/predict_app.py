from flask import Flask,render_template,request, jsonify
import base64
import numpy as np
from keras.models import load_model
import tensorflow as tf
import io
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from flask import request
from flask import jsonify
from flask import Flask
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image

app= Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/about/')
def about(): #next page.. localhost:5000/about
   return render_template("second.html")

@app.route('/vgg16/')
def vgg16(): #next page.. localhost:5000/about
    return render_template("vgg_predict.html")

@app.route('/Keras_Model/')
def Keras_Model(): #next page.. localhost:5000/about
    return render_template("Keras_Model.html")


@app.route('/CNN_Model/')
def CNN_Model(): #next page.. localhost:5000/about
    return render_template("cnnModel.html")

def get_model():
     global model,graph
     model=load_model('weight/new_model_cnn.h5')
     print("Model loaded")
def preprocess_image(image,target_size):
    image=image.resize(target_size)
    image=img_to_array(image)
    image= np.expand_dims(image,axis=0)

    return image

print(" * Loading Keras Model")
get_model()
@app.route("/predict",methods=["POST"])

def predict():
    message=request.get_json(force=True)
    encoded=message['image']
    decoded=base64.b64decode(encoded)
    image=Image.open(io.BytesIO(decoded))
    processed_image= preprocess_image(image,target_size=(64,64))
    rslt = model.predict(processed_image)
    if rslt[0][0] == 1:
        prediction = "buildings"
    elif rslt[0][1]==1:
        prediction = "forest"
    elif rslt[0][2]==1:
        prediction = "glacier"
    elif rslt[0][3]==1:
        prediction = "mountain"
    elif rslt[0][4]==1:
        prediction = "sea"
    elif rslt[0][5]==1:
        prediction = "street"
    #prediction="Prachi"
    response={
        'prediction':
        { 'prediction' : prediction
          }
        }
    return jsonify(response)
print("working............")



def get_model1():
     global classifier,graph
     classifier=load_model('weight/vgg_16.h5')
     print("Model loaded")
     
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model

# prepare the image for the VGG model
def preprocess_VGGimage(image,target_size):
    image=image.resize(target_size)
    image=img_to_array(image)
    image=image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    return image
print("working111............")

get_model1()
@app.route("/vgg_predict",methods=["POST"])
def vgg_predict():
    message=request.get_json(force=True)
    encoded=message['image']
    decoded=base64.b64decode(encoded)
    image=Image.open(io.BytesIO(decoded))
    processed_image= preprocess_VGGimage(image,target_size=(224,224))

    yhat = classifier.predict(processed_image)
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    rslt= ('%s (%.2f%%)' % (label[1], label[2]*100))
    response={
        'prediction':
        { 'prediction' : rslt
          }
        }
    return jsonify(response)

if __name__=="__main__":
    app.run(debug=True)
     
          
