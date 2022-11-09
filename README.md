# store_sales
Prodict the store sales states of the future based on mathine learning.

**数据准备**

将train与transactions数据集放到/dataset/train下

将test数据集放到/dataset/test下

其余数据集放到/dataset/extra下

如有变动，可在配置文件config,yml中修改

**预处理**

/pretreatment下包括数据预处理的全部脚本

包括：
- dataload: 数据加载
- datamachine: 数据处理
- datavisualize: 数据可视化[数据分析]

**运行**

运行run.py开始进行数据预处理和训练
