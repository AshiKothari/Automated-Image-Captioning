from flask import Flask, render_template, redirect, request
import caption_it

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/about')
def about():
    return "<h1> Hi this is Ashi! and this project is Image Captioning. Hope you enjoy!</h1>"

@app.route('/home')
def home():
    return redirect('/')

@app.route('/', methods = ['POST'])
def marks():
    if request.method == 'POST':
        f = request.files['userfile']
        path = "./static/{}".format(f.filename)
        f.save(path)
        caption = caption_it.captionis(path)
        dic = {'image' : path ,
               'caption' : caption}
    return render_template("index.html",cap=dic)

if __name__ == '__main__':
    app.run(debug=True)
    #app.run("0.0.0.0", 5000, threaded=False)
