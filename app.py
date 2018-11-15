from flask import Flask
# from our thing
from lstm1 import run_lstm
from knn import run_knn

app = Flask(__name__)

@app.route('/lstm', methods=['POST'])
def lstm():
    output = run_lstm()
    return output


@app.route('/knn', methods=['POST'])
def knn():
    output = run_knn()
    return output

if __name__ == '__main__':
    app.run('0.0.0.0', port=3333) # inherit whatever IP of the server that's it's running on
