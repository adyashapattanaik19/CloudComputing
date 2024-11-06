import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , request, render_template
#from gevent.pywsgi import WSGIServer
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


app = Flask(__name__)
model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        
        #print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        preds = model.predict(x)
        
        print("prediction",preds)
            
        index = ['No DR','Mild','Moderate','Severe','Proliferative DR']
        
        print(np.argmax(preds))
        
        text = "The Stage of Diabetic Retinopathy is : " + str(index[np.argmax(preds)])
        
    return text
if __name__ == '__main__':
    app.run(debug = False, threaded = False)
        
        
        
    
    
    