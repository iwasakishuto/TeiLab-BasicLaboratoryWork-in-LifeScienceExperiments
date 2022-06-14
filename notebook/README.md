## Notebooks

| SubDirectory   | Notebook                                                                                                                                                                                                                                          | Description                                                                                                                                    |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|

| Colaboratory   | [ `Colaboratory/microarray2021S.ipynb` ](https://nbviewer.jupyter.org/github/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/blob/main/notebook/Colaboratory/microarray2021S.ipynb)                                               | 2021年 Sセメスター （１週目）で用いた講義資料。データの読み込みにパッケージを多用しすぎてわかりづらくなってしまった。 [![Open in Colab](https://img.shields.io/badge/1st%20week-open%20in%20Colab-3d80bc?style=flat-radius&logo=google-colab)](https://colab.research.google.com/drive/1_Hr0acMj2vXyKzuCl3MiD6Lede-Vsydv?usp=sharin)  |
| Colaboratory   | [ `Colaboratory/microarray2021S_2nd.ipynb` ](https://nbviewer.jupyter.org/github/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/blob/main/notebook/Colaboratory/microarray2021S_2nd.ipynb)                                       | 2021年 Sセメスター （２週目）で用いた講義資料。よりインタラクティブな講義を目指した。 [![Open in Colab](https://img.shields.io/badge/2nd%20week-open%20in%20Colab-3d80bc?style=flat-radius&logo=google-colab)](https://colab.research.google.com/drive/1_Hr0acMj2vXyKzuCl3MiD6Lede-Vsydv?usp=sharin)
   |
| Colaboratory   | [ `Colaboratory/microarray2020A_py.ipynb` ](https://nbviewer.jupyter.org/github/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/blob/main/notebook/Colaboratory/microarray2020A_py.ipynb)                                         | 2020年 Aセメスター で用いた講義資料                                                                                                            |
| Supplementary  | [ `Supplementary/Add-Anotation-Data.ipynb` ](https://nbviewer.jupyter.org/github/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/blob/main/notebook/Supplementary/Add-Anotation-Data.ipynb)                                       | 2021年 Sセメスターから使用したデータにアノテーションデータを追加する方法を記載。（プレートスキャンを行った会社から送られてきた資料を用いる。） |
| Supplementary  | [ `Supplementary/Data-Details.ipynb` ](https://nbviewer.jupyter.org/github/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/blob/main/notebook/Supplementary/Data-Details.ipynb)                                                   | マイクロアレイデータのヘッダー部分から前処理等を読み解き、データの詳細な理解を目指す。                                                         |
| Local          | [ `Local/Start-JupyterNotebook-with-Poetry-Environment.ipynb` ](https://nbviewer.jupyter.org/github/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/blob/main/notebook/Local/Start-JupyterNotebook-with-Poetry-Environment.ipynb) | LocalでPoetryを使って環境構築をした際に、その環境をJupyter Notebookで使うための方法を記載。                                                    |
| Local          | [ `Local/Main-Lecture-Material-plotly.ipynb` ](https://nbviewer.jupyter.org/github/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/blob/main/notebook/Local/Main-Lecture-Material-plotly.ipynb)                                   | ローカル環境で、パッケージをフルに使って解析を行う。可視化には plotly を用いている。                                                           |
| Local          | [ `Local/Main-Lecture-Material-matplotlib.ipynb` ](https://nbviewer.jupyter.org/github/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/blob/main/notebook/Local/Main-Lecture-Material-matplotlib.ipynb)                           | ローカル環境で、パッケージをフルに使って解析を行う。可視化には matplotlib を用いている。                                                       |

<details>
  <summary>How to generate this table</summary>

```python
  import os
  from tabulate import tabulate
  tabular_data = []
  PREFIX = "https://nbviewer.jupyter.org/github/iwasakishuto/TeiLab-BasicLaboratoryWork-in-LifeScienceExperiments/blob/main/notebook"
  for subdir in os.listdir():
      if (not os.path.isdir(subdir)) or subdir.startswith("."): continue
      for fn in os.listdir(subdir):
          if not fn.endswith(".ipynb"): continue
          fp = f"{subdir}/{fn}"
          link = f"{PREFIX}/{fp}"
          tabular_data.append([subdir, f"[`{fp}`]({link})", ""])
  print(tabulate(tabular_data=tabular_data, headers=["SubDirectory", "Notebook", "Description"], tablefmt="github"))
  ```

</details>
