from flask import Flask, render_template

app = Flask(__name__,static_url_path='/static')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Model/')
def Model():
    return render_template('Model.html')
@app.route('/discord/')
def discord():
    return render_template('discord.html')
@app.route('/modelresults/')
def modelresults():
    return render_template('modelresults.html')

@app.route('/predict', methods = ['POST'])
def predict(): 
    return render_template('result.html')   

if __name__ == "__main__":
    app.run(debug=True)


