from flask import Flask, make_response, render_template, request
from urllib.parse import parse_qs

app = Flask(__name__)


@app.route('/')
def hello_world():
    res = make_response(render_template("index.html"))
    return res


@app.route("/Ratings")
def Ratings():
    """
    在这里，需要ISBN号、以及其封面、作者名、书名(都是列表)
    :return:
    """
    ISBNList = ['0886774632', '050552600X', '1558686274', '0440223202', '0312966776']
    res = make_response(render_template("Ratings.html",
                                        Acccc=range(len(ISBNList)),
                                        ISBNList=ISBNList,
                                        fengMian=["http://images.amazon.com/images/P/0425176428.01.LZZZZZZZ.jpg"],
                                        Title="IF you",
                                        Author="AAAA"))
    return res


@app.route("/submitinfo", methods=["GET", "POST"])
def submitinfo():
    if request.method == "GET":
        res = make_response(render_template("Ratings.html"))
        return res
    elif request.method == "POST":
        # print(type(request.files))
        # print(request.files.items())
        # print(request.files.keys())
        # for i in request.files.keys():
        #     print(i)
        # res = request.data
        # print("res：".format(res))
        data = request.get_data()
        print(data)
        datadict = parse_qs(data)
        print(datadict)
        # data = json.loads(data)
        # print(data)
        return "成功了"
    else:
        return "error"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
