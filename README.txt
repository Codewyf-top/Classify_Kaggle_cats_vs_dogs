1.数据集： 把train按照train:test=7:3的比例划分。数据集情况：train(17500张)，test(7500张)。
2.数据预处理：代码在utils文件中的utils.py，已经备注地方。包括调整图片大小，水平翻转，一定角度旋转，添加噪声，图片归一化
3.整个工程用到的代码：train.py, test.py, model文件夹中的RESNET.py, src文件夹中的config.py, utils文件中的utils.py.其他的使用不到
4代码运行：
训练： python train.py --path ./checkpoints/1/
(注：./checkpoints/1/文件夹下面的config.yml里面是超参数，可改动)

测试： 训练生成的模型在./checkpoints/1/model里面，找到最新的含有best的模型。把模型地址写到test.py的修改点处。
运行： python test.py --path ./checkpoints/1/
tensorboard：
运行：tensorboard --logdir runs
