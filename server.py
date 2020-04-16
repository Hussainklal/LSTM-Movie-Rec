from flask import Flask
from flask import render_template
from flask import jsonify
import os
import json
import opts
import torch
import pickle
import numpy as np
from utils import *
from Model.RNN import Model as torch_rnn
from Model.transformer.model import Model as torch_transformer

from movies import LSTM_Movie_Rec

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/results/<params>', methods=['GET'])
def get_results(params):
    print("Get results called", params)
    movie_list = json.loads(params)
    print(movie_list)
    inp_seq = list(map(lambda x:MOVIE_TO_ID[x],movie_list))
    result = trans_predict(transformer_model,inp_seq,opt,MOVIE_TO_ID,ID_TO_MOVIE)
    print(result)
    # result = rnn_predict(rnn_model,inp_seq,opt,MOVIE_TO_ID,ID_TO_MOVIE)
    # global my_model
    # result = my_model.top_k_movies(movie_list)
    return jsonify(result)


my_model = None

if __name__ == "__main__":
    opt = opts.parse_opt()
    opt = vars(opt)

    ML_1M_PATH = "./data/ml-1m/ratings.dat"
    with open("./data/movie_to_id.pkl","rb") as f:
        MOVIE_TO_ID = pickle.load(f)

    with open("./data/movie_map.pkl","rb") as f:
        ID_TO_MOVIE = pickle.load(f)

    transformer_model = torch_transformer(num_users=None,
                  num_items=len(MOVIE_TO_ID),
                  opt=opt)

    transformer_model.load_state_dict(torch.load("./save/model_transformer_140.pth",map_location=torch.device('cpu')))
    transformer_model.eval()

    rnn_model = torch_rnn(num_users=None,
                        num_items=len(MOVIE_TO_ID),
                        opt=opt)

    # rnn_model.load_state_dict(torch.load("./save/model_rnn_10_70.pth",map_location=torch.device('cpu')))

    Rec = LSTM_Movie_Rec(path=ML_1M_PATH, sep='::')
    Rec.load_model()
    my_model = Rec

    # Run server
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
