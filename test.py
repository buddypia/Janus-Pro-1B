
from flask import Flask
app = Flask(__name__)

# http://127.0.0.1:5000をルートとして、("")の中でアクセスポイント指定
# @app.route("hoge")などで指定すると、http://127.0.0.1:5000/hogeでの動作を記述できる。
@app.route("/")
def hello():
    return "Hello World!"

if __name__ == "__main__":
    # webサーバー立ち上げ
    app.run()