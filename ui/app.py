import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request
from pl_predictor import load_all_seasons, PLPredictor

app = Flask(__name__)

print("Loading model (this takes ~30s)...")
_df = load_all_seasons()
_predictor = PLPredictor()
_predictor.fit(_df)
print("Model ready.")

@app.route('/')
def index():
    return render_template('index.html', teams=_predictor.teams)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    home = data.get('home', '').strip()
    away = data.get('away', '').strip()
    if not home or not away:
        return jsonify({'error': 'Please select both teams.'}), 400
    if home == away:
        return jsonify({'error': 'Home and away teams must be different.'}), 400
    try:
        result = _predictor.predict(home, away)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(debug=False, port=port)
