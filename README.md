## Implementation-of-VGCN-BERT-for-Rumor-Detection

> Further work about project "Improving Rumor Detection with User Comments". We use **VGCN** and **BERT** to extract global and local features from texts and apply **Attention Mechanism** for feature fusion to improve rumor detection performance even further.

* [Previous Project](https://github.com/oraccc/Improving-Rumor-Detection-with-User-Comments)

### 1. Project Introduction

* **Overall Structure**

<img src="https://raw.githubusercontent.com/oraccc/Implementation-of-VGCN-BERT-for-Rumor-Detection/master/images/overall-structure.png" width="500" />



* **Model Structure** 

  * :one: **Segmentation Rumor Text Feature Extraction Module** （分段谣言文本特征提取模块）

    <img src="https://raw.githubusercontent.com/oraccc/Implementation-of-VGCN-BERT-for-Rumor-Detection/master/images/module1.png" width="475" />

  * :two: **Rumor Features Fusion and Classification Module** （谣言特征融合与分类模块）

    <img src="https://raw.githubusercontent.com/oraccc/Implementation-of-VGCN-BERT-for-Rumor-Detection/master/images/module2.png" width="500" />

---

### 2. Usage

* Clean Redundant Data

* Format dataset from **json** to **csv**

  <img src="https://raw.githubusercontent.com/oraccc/Implementation-of-VGCN-BERT-for-Rumor-Detection/master/images/dataset-reorganize.png" width="500" />

#### :one: data_reorganize_pheme.ipynb / data_reorganize_weibo.ipynb

* Dataset reorganize & segment long text

  <img src="https://raw.githubusercontent.com/oraccc/Implementation-of-VGCN-BERT-for-Rumor-Detection/master/images/long-text-segment.png" width="500" />

#### :two: **prepare_data.py**

* Get Vocabulary Graph

  <img src="https://raw.githubusercontent.com/oraccc/Implementation-of-VGCN-BERT-for-Rumor-Detection/master/images/graph-construction.png" width="350" />

#### :three: train_vgcn_bert.ipynb

* Train VGCN_BERT on graph data and comments data

#### :four: joint_vgcn_bert_model.ipynb

* Feature concatenate

  <img src="https://raw.githubusercontent.com/oraccc/Implementation-of-VGCN-BERT-for-Rumor-Detection/master/images/feature-concatenate.png" width="500" />

* Apply trained VGCN_BERT model to get rumor detection results
