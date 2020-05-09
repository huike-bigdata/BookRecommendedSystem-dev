import random

from flask import Flask, make_response, render_template, request
from urllib.parse import parse_qs
from src import BookRecommended
import pandas as pd

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
    # 随机选择几本书让用户评分
    global BR
    newbooklist = random.sample(BR.ISBN_list, 5)
    # 获取书籍详细的信息
    bookInfo = BR.getBookInfo(booksInfo, newbooklist)
    print("呈现给用户的书籍的ISBN为：{}".format(bookInfo[3]))
    print("呈现给用户的书籍的名字为：{}".format(bookInfo[0]))
    print("呈现给用户的书籍的作者为：{}".format(bookInfo[1]))
    print("呈现给用户的书籍的封面为：{}".format(bookInfo[2]))

    ISBNList = bookInfo[3]
    res = make_response(render_template("Ratings.html",
                                        Acccc=range(len(ISBNList)),
                                        ISBNList=ISBNList,
                                        fengMian=bookInfo[2],
                                        Title=bookInfo[0],
                                        Author=bookInfo[1]))
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
    print("********************************************执行了{}".format("__name__ == '__main__'"))
    BR = BookRecommended.BookBookRecommended(K=5)
    # 加载配置文件
    with open("src/conf.json", "r") as f:
        conf = eval(f.read())
    # 加载书籍信息文件
    booksInfo = pd.read_csv("./data/BX-Books.csv", sep=";", encoding="utf-8")

    # 加载模型（内部会根据配置文件选择重新训练还是加载已有的模型）
    BR.loadModel(conf["TrainingOrLoad"])
    app.run(host="0.0.0.0", port=80)

