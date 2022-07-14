# 当代人工智能项目5：多模态情绪分析

Author: GONGGONGJOHN

本仓库为课程《当代人工智能》最后一个课程项目的代码仓库，全部代码均可执行、可复现。

## 环境依赖

本项目使用Python3.8开发完成，相关依赖模块已导出至`requirements.txt`。使用如下命令即可直接安装所有依赖包：

```shell
pip -r requirements.txt
```

## 目录结构

```
.
├── README.md
└── data_utils.py
```

## 运行实验

若要使用原数据集运行实验，请将图像/文本数据放在`data/source/`中。由于原数据集中包含非utf-8/ANSI编码文本，请先使用GBK编码打开并将其翻译为英语再进行读取。