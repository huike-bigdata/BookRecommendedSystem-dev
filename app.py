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
    global ISBNList
    newbooklist = random.sample(BR.ISBN_list, 5)
    # 获取书籍详细的信息

    bookInfo = BR.getBookInfo(booksInfo, newbooklist)
    print("呈现给用户的书籍的ISBN为：{}".format(bookInfo[3]))
    print("呈现给用户的书籍的名字为：{}".format(bookInfo[0]))
    print("呈现给用户的书籍的作者为：{}".format(bookInfo[1]))
    print("呈现给用户的书籍的封面为：{}".format(bookInfo[2]))

    ISBNList = bookInfo[3]
    # ISBNList:用户打分的书籍列表
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
        # 添加新用户到模型中
        global ISBNList
        BR.addRatings(ISBNList, datadict[b"Ratings"], user_id=88)

        print("{} — 开始预测".format(BookRecommended.nowTime()))
        # 预测
        BR.LFM_grad_desc(max_iter=conf["max_iter"], alpha=0.001, lamda=0.002)

        print(BR.cost)  # 输出损失函数

        # print(BRS.predR)  # 输出预测值

        # 取出单用户预测的TopN评分
        global ISBN_topN, Rating_topN
        ISBN_topN, Rating_topN = BR.getTopRatings(topnum=5, ISBNS=ISBNList, user_id=88)
        print("{} — 预测完成".format(BookRecommended.nowTime()))
        print("推荐您阅读：{}".format(ISBN_topN), Rating_topN)
        # res = make_response(render_template("Ratings.html"))
        return "提交成功，<a href=\"/showRecommended\">跳转</a>"
    else:
        return "error"


@app.route("/showRecommended")
def showRecommended():
    bookInfo = BR.getBookInfo(booksInfo, ISBN_topN, suijichou=False)
    print("呈现给用户的书籍的ISBN为：{}".format(bookInfo[3]))
    print("呈现给用户的书籍的名字为：{}".format(bookInfo[0]))
    print("呈现给用户的书籍的作者为：{}".format(bookInfo[1]))
    print("呈现给用户的书籍的封面为：{}".format(bookInfo[2]))

    showISBNList = bookInfo[3]
    # ISBNList:展示给用户的推荐书籍
    res = make_response(render_template("Recommended.html",
                                        Acccc=range(len(showISBNList)),
                                        ISBNList=showISBNList,
                                        fengMian=bookInfo[2],
                                        Title=bookInfo[0],
                                        Author=bookInfo[1]))
    return res
    pass


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
