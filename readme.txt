

0.LCCTV  DATA 解压到同一目录


1.修改数据路径

LCCTV/lib/test/evaluation/local.py

settings.my_path = '/home/lbz/dz/DATA'
settings.prj_dir = '/home/lbz/dz/LCCTV'
settings.save_dir = '/home/lbz/dz/LCCTV/output'


2.修改数据路径

LCCTV/lib/test/evaluation/mydataset.py

{'name': 'real1', 'path': '1/imgs', 'startFrame': 1, 'endFrame': 100, 'nz': 8,
    'ext': 'jpg',
    'anno_path': '1/groundtruth.txt',
    'object_class': 'real1'},
{'name': 'real1', 'path': '2_all/img', 'startFrame': 1, 'endFrame': 100, 'nz': 8,
    'ext': 'jpg',
    'anno_path': '2_all/groundtruth.txt',
    'object_class': 'real1'},
{'name': 'real1', 'path': '3/img', 'startFrame': 1, 'endFrame': 100, 'nz': 8,
    'ext': 'jpg',
    'anno_path': '3/groundtruth.txt',
    'object_class': 'real1'},


3.安装环境依赖

cd LCCTV

conda env create -f grm.yml


4.运行

4.1删除老结果

rm LCCTV/output/lcctv/B9_cae_center_all_ep300_300/*


4.2运行
export LD_LIBRARY_PATH=/usr/lib/wsl/lib/:$LD_LIBRARY_PATH
python tracking/test.py lcctv B9_cae_center_all_ep300 --dataset my --threads 2 --num_gpus 1 --ep 300


4.3结果（出现如下打印代表完成）

Tracker: lcctv B9_cae_center_all_ep300 300 ,  Sequence: real1
Tracker: lcctv B9_cae_center_all_ep300 300 ,  Sequence: real1
地震烈度计算结果 - 序列: real1
烈度指数(Ii): 4.6, PGA: 0.1466306916224067, PGV: 0.029326135967748912
地震烈度计算结果 - 序列: real1
最大X位移: 0.0014995879999999961m, 最大Y位移: 0.0053116529999999995m
烈度指数(Ii): 4.3, PGA: 0.11780082511859227, PGV: 0.02593015140285019
最大X位移: 0.0035028919999999996m, 最大Y位移: 0.005825071000000015m






