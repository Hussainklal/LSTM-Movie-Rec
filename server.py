from flask import Flask
from flask import render_template
from flask import jsonify
import os
import json

from movies import LSTM_Movie_Rec

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/results/<params>', methods=['GET'])
def get_results(params):
    print("Get results called", params)
    movie_list = json.loads(params)
    global my_model
    result = my_model.top_k_movies(movie_list)
    return jsonify(result)


my_model = None

if __name__ == "__main__":
    ML_1M_PATH = "./data/ml-1m/ratings.dat"
    Rec = LSTM_Movie_Rec(path=ML_1M_PATH, sep='::')
    Rec.load_model()
    my_model = Rec

    # Run server
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
