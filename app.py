from flask import Flask, request, jsonify
from my_functions import add_numbers, greet  # 準備した関数をインポート

app = Flask(__name__)

@app.route('/add', methods=['GET'])
def add():
    try:
      x = int(request.args.get('x'))
      y = int(request.args.get('y'))
      result = add_numbers(x, y)
      return jsonify({'result': result})
    except (TypeError, ValueError):
      return jsonify({'error': 'Invalid input. Please provide valid integers for x and y.'}), 400


@app.route('/greet', methods=['POST'])
def greet_user():
    data = request.get_json()  # リクエストボディからJSONデータを取得
    if data and 'name' in data:
        name = data['name']
        greeting = greet(name)
        return jsonify({'greeting': greeting})
    else:
        return jsonify({'error': 'Name is required.'}), 400
    

if __name__ == '__main__':
    app.run(debug=True)  # 開発中はdebug=Trueにしておくと便利
