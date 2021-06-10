# 生命科学基礎実験

シークエンシング、生化学反応シミュレーション、および遺伝子発現プロファイリング等の生物情報科学的実験を行う。本実習は、生物化学科と合同で実施する。

![程研究室](https://camo.githubusercontent.com/207bfad0a9138764bcfb51e04353f209de425aaf201648b706936e255b749dd6/687474703a2f2f75692d7465692e726e61692e6a702f6173736574732f696d616765732f746f702d3230303830372e6a7067)
[![GitHub version](https://badge.fury.io/gh/iwasakishuto%2FTeiLab-BasicLaboratoryWork-in-LifeScienceExperiments.svg)](https://github.com/iwasakishuto/Translation-Gummy)
[![Install This Package](https://github.com/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/workflows/Install%20This%20Package/badge.svg)](https://github.com/iwasakishuto/Translation-Gummy/blob/master/.github/workflows/execute_python_package.yml)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/blob/master/LICENSE)
[![実習wiki](https://img.shields.io/badge/website-wiki-9ecca4)](https://elb.translation-gummy.com/)
[![程研HP](https://img.shields.io/badge/website-HP-9ecca4)](https://elb.translation-gummy.com/)
[![Documentation](https://img.shields.io/badge/Documentation-portfolio-1f441e?style=flat-radius)](https://iwasakishuto.github.io/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/index.html)
[![twitter badge](https://img.shields.io/badge/twitter-Ask%20Author-1da1f2?style=flat-radius&logo=twitter)](https://www.twitter.com/messages/compose?recipient_id=1042783905697288193&text=2021年度の生命科学基礎実験を受講しているものですが、)

## 講義基本情報

|時間割／共通科目コード|コース名|教員|学期|時限|
|:-:|:-:|:-:|:-:|:-:|
|0560527<br>FSC-BI3B07P1|生命科学基礎実験|程　久美子|S2|火曜3限、火曜4限 他|

|教室|単位|他学部履修|講義使用言語|実務経験のある教員による授業科目|開講所属|
|:-:|:-:|:-:|:-:|:-:|:-:|
|理学部３号館 015|3|不可|日本語|NO|理学部|

※ スケジュール等は[実習wiki](http://ui-tei.rnai.jp/microarray/doku.php)や[このRepositoryのwiki](https://github.com/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/wiki)を参照して下さい

## 使い方

環境構築の容易さという観点から、基本的には[Google Colab]()を用いて講義を行いますが、

- [🤗 ドキュメントを確認したい。](#-ドキュメントを確認したい)
- [👍 自分のローカル環境で実行したい。](#-自分のローカル環境で実行したい)
- [👏 Pull Requestを送りたい](#-pull-requestを送りたい)

といった方々向けに使い方を簡単に記載します。

### 🤗 ドキュメントを確認したい

1. 右上の \[Code\] というボタンから Download ZIP を押して、ZIPファイルをダウンロードします。
2. ダウンロードしたフォルダ内の `docs` -> `index.html` をクリックして、お好きなブラウザ（Google Chromeを推奨）で開いて下さい。以下のような画面が現れると思います。

![Documentation](images/documentation.png)

### 👍 自分のローカル環境で実行したい

- [Pyenv](https://github.com/pyenv/pyenv#installation)：異なるバージョンのpython環境を柔軟に構築
- [Poetry](https://github.com/python-poetry/poetry#installation)：パッケージのバージョンや依存関係を記録

を用いて、実行環境を作成することを推奨します。また、作成した環境でローカルのJupyter Notebookを起動したい場合は、<a href="https://nbviewer.jupyter.org/github/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/blob/main/notebook/%5Blocal%5D%20Start%20Jupyter%20Notebook%20with%20Poetry%20Environment.ipynb">`[local] Start Jupyter Notebook with Poetry Environment.ipynb`</a>を参照して下さい。


```sh
$ git clone https://github.com/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments.git
$ cd TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments
$ pyenv install 3.8.9
$ pyenv local 3.8.9
$ python -V
Python 3.8.9
$ poetry install
```

<b>※ わからないことがあれば適宜質問して下さい。</b>

### 👏 Pull Requestを送りたい

一応テンプレートを用意していますが、好きなように送って下さい☺️