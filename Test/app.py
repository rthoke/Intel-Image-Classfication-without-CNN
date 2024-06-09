from flask import *  
import os
from scipy import *
import cv2
import mahotas
import joblib
import numpy as np
from werkzeug.utils import secure_filename
from xgboost import XGBClassifier


app = Flask(__name__)  


UPLOAD_FOLDER = "Save_images"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'This is your secret key to utilize session'
@app.route('/')  
def upload():  
    return render_template("Upload.html")  


classes = ['Building', 'Forest', 'Glacier',  'Mountains','Sea', 'Streets']

@app.route('/success', methods = ['POST','GET'])  
def success():  
    if request.method == 'POST':
        file = request.files['sile']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        imganeme = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        x_image = []
        image = cv2.imread(imganeme)
        image_resized = cv2.resize(image, (300,300))
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        image_hu = cv2.HuMoments(cv2.moments(image_gray)).flatten()
        image_har = mahotas.features.haralick(image_gray).mean(axis=0)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_hist  = cv2.calcHist([image_hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        image_hist_flat = image_hist.flatten()
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(image_resized, None)
        if des is not None:
            sift_features = des.flatten()
            if len(sift_features) < 128:
                sift_features = np.pad(sift_features, (0, 128 - len(sift_features)), mode='constant')
            else:
                sift_features = sift_features[:128]
        else:
          sift_features = np.zeros(128)
        f_vector_concat = np.hstack([image_hist_flat, image_har, image_hu,sift_features])
        x_image.append(f_vector_concat)
        x_image_n = np.array(x_image)

        clf2 = joblib.load('GradientBoostingClassifier.pkl')
        preds2 = clf2.predict(x_image_n.reshape(1,-1))
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template("Upload.html",name = classes[preds2[0]])  

  
if __name__ == '__main__':  
    app.run(host='0.0.0.0',port = '2221')  
