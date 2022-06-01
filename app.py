import keras
from keras import backend as K
import modelCore
import tensorflow as tf
from cv2 import cv2
import numpy as np
from flask import Flask, redirect, request, render_template
import matplotlib.pyplot as plt
import base64
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

app = Flask(__name__)
cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:port"}})

#Load ManTraNet
manTraNet_root = 'ManTraNet/'
manTraNet_srcDir = os.path.join( manTraNet_root, 'src' )
sys.path.insert( 0, manTraNet_srcDir )
manTraNet_modelDir = os.path.join( manTraNet_root, 'pretrained_weights' )

from ManTraNet.src import modelCore
model_mantra = modelCore.load_pretrain_model_by_index( 4, manTraNet_modelDir )

def read_rgb_image( image_file ) :
    rgb = cv2.imread( image_file, 1 )[...,::-1]
    return rgb
    
def decode_an_image_array( rgb, manTraNet ) :
    x = np.expand_dims( rgb.astype('float32')/255.*2-1, axis=0 )
    t0 = datetime.now()
    with graph.as_default():
        y = manTraNet.predict(x)[0,...,0]
    t1 = datetime.now()
    return y, t1-t0

def decode_an_image_file( image_file, manTraNet ) :
    rgb = read_rgb_image( image_file )
    rgb = cv2.resize(rgb, dsize=(640,480), interpolation=cv2.INTER_CUBIC)
    mask, ptime = decode_an_image_array( rgb, manTraNet )
    return rgb, mask, ptime.total_seconds()
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def base():
    if request.method == 'GET':
        return render_template("base.html", output=0)
    else:
        if 'input_image' not in request.files:
            print("No file part")
            return redirect(request.url)

        file = request.files['input_image']

        if file.filename == '':
            print('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            inp_img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            decode_an_image_file(inp_img)
            output = cv2.imread('h.png')
            _, outputBuffer = cv2.imencode('.jpg', output)
            OutputBase64String = base64.b64encode(outputBuffer).decode('utf-8')
            return render_template("base.html", img=OutputBase64String, output=1)
        
if __name__ == '__main__':
    app.run()
