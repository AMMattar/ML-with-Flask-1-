from flask import Flask, request, render_template
from joblib import load

app = Flask(__name__)
pipeline = load('my_ML.joblib')

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == "POST":
        text = [request.form['search']]
        response = pipeline.predict(text)
        return render_template('answer.html', response=response)
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)