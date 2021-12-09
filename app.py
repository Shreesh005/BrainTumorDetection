# IMPORTING NECESSARY LIBRARIES
from flask import Flask, render_template, request
import os 
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# LOAD THE MODEL
model = load_model("model/model.h5")

# WRITING THE FUNCTION 
def braintumor(model, image):
    test_image = load_img(image, target_size=(200,200))
    test_image = img_to_array(test_image)/255
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image).round(3)
    prediction = np.argmax(result)

    if prediction==0:
        message = "By the MRI it appears that the person doesn't has brain tumor. Your tumor report is negative"
    else:
        message = "By the MRI it appears that the person is suffering from brain tumor. So it is advisable to contact a doctor."

    return message
# CREATING FLASK INSTANCE
app = Flask(__name__)

# CREATING END POINTS
@app.route('/',methods=['GET','POST'])
def home():
    return render_template("index.html")
@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['uploadfile']
        filename = file.filename
        filepath = os.path.join('static/userUpload', filename)
        file.save(filepath)
        final = braintumor(model,image=filepath)
        return render_template('prediction.html',user_image= filepath, finaloutput = final)
if __name__=='__main__':
    app.run(debug=True)
