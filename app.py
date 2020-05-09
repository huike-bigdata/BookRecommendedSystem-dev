from flask import Flask, make_response, render_template

app = Flask(__name__)


@app.route('/')
def hello_world():
    res = make_response(render_template("index.html"))
    return res


@app.route("/Ratings")
def Ratings():
    """
    在这里，需要ISBN号、以及其封面、作者名、书名
    :return:
    """
    res = make_response(render_template("Ratings.html",
                                        Acccc=range(10),
                                        ISBN="11111",
                                        fengMian=["http://images.amazon.com/images/P/0425176428.01.LZZZZZZZ.jpg"],
                                        Title="IF you",
                                        Author="AAAA"))
    return res


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
