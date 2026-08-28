#!/usr/bin/env python
# coding: utf-8

# In[10]:


# 解压数据
get_ipython().system(' ls ./data')
# ! unzip -q -o ./data/data264550/image_set.zip -d ./data
# ! unzip -q -o <lane_train_archive>.zip -d ./data
# ! unzip -q -o <lane_eval_archive>.zip -d ./data


# In[11]:


import cv2, os, glob
# 配置相关和获取函数
train_cfg = {
    "train_paths": [["./data/image_set_lane","data.json"], ],
    "eval_paths": [["./data/image_set_lane_eval","data.json"], ],
    "num_workers": 0,
    "infer_dir": "./model/cnn23/inference",
    "model_save_dir": "./model/cnn23/dymic",
    "model_params_name": "cnn_lane.pdparams",
    "model_opt_name": "cnn_lane.pdopt",
    "model_checkpoint_name": "cnn_lane.pkl",
    "train_batch_size": 256,
    "eval_batch_size": 256,
    "lr": 0.001,
    "epochs": 200,
    "input_size": [128, 128],
}

def get_data_paths():
    # 数据集路径
    train_paths = train_cfg["train_paths"]
    eval_paths = train_cfg["eval_paths"]
    return train_paths, eval_paths

def get_last_model_name():
    import glob
    model_save_dir = train_cfg["model_save_dir"]
    models_list = glob.glob(model_save_dir + "/*")
    models_list.sort()
    if len(models_list) == 0:
        raise RuntimeError("no model to infer, at " + model_save_dir)
    last_name = os.path.split(models_list[-1])[-1]
    return last_name

def get_model_path(file_name):
    # 模型保存路径
    model_save_dir = train_cfg["model_save_dir"]
    model_params_path = os.path.join(model_save_dir, file_name, train_cfg["model_params_name"])
    model_opt_path = os.path.join(model_save_dir, file_name, train_cfg["model_opt_name"])
    checkpoint_path = os.path.join(model_save_dir, file_name, train_cfg["model_checkpoint_name"])
    return model_params_path, model_opt_path, checkpoint_path

def get_infer_path(file_name):
    # 模型保存路径
    infer_save_dir = train_cfg["infer_dir"]
    infer_save_path = os.path.join(infer_save_dir, file_name, "cnn_lane")
    return infer_save_path


# In[12]:


# 导入paddle相关库
import paddle
from paddle import nn


# In[13]:


# 定义巡航模型
class CnnModel(nn.Layer):
    # 定义模型
    def __init__(self):
        super(CnnModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2D(3, 32, 5, stride=2),
            nn.ReLU(),

            nn.Conv2D(32, 32, 5, stride=2),
            nn.ReLU(),

            nn.Conv2D(32, 64, 5, stride=2),
            nn.ReLU(),

            nn.Conv2D(64, 64, 3, stride=2),
            nn.ReLU(),

            nn.Conv2D(64, 128, 3, stride=1),
            # nn.BatchNorm(32),
            nn.ReLU(),

            nn.Dropout(p=0.1),

            nn.Conv2D(128, 128, 3, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Flatten(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 2),
            # nn.Tanh()
        )

    def forward(self, inputs):
        x = self.features(inputs)
        return x

from paddle.io import Dataset, DataLoader, ComposeDataset
from paddle import to_tensor
import os, json
import random
import cv2
import numpy as np
from PIL import Image

# 定义数据增强手段
def color_filter_autumn(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_AUTUMN)
    return im_color


def color_filter_bone(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_BONE)
    return im_color


def color_filter_winter(img):
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_WINTER)
    return im_color


def apply_hue(img):

    low, high, prob = [-18, 18, 0.5]
    if np.random.uniform(0., 1.) < prob:
        return img


    delta = np.random.uniform(low, high)
    u = np.cos(delta * np.pi)
    w = np.sin(delta * np.pi)
    bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
    tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                     [0.211, -0.523, 0.311]])
    ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                      [1.0, -1.107, 1.705]])
    t = np.dot(np.dot(ityiq, bt), tyiq).T
    img = np.dot(img, t)
    img = np.array(img).astype(np.uint8)
    return img


def apply_saturation(img):
    low, high, prob = [0.5, 1.5, 0.5]
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)

    gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
    gray = gray.sum(axis=2, keepdims=True)
    gray *= (1.0 - delta)
    img *= delta
    img += gray
    img = np.array(img).astype(np.uint8)
    return img


def apply_contrast(img):
    low, high, prob = [0.5, 1.5, 0.5]
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)

    img *= delta
    img = np.array(img).astype(np.uint8)
    return img


def apply_brightness(img):
    low, high, prob = [0.5, 1.5, 0.5]
    if np.random.uniform(0., 1.) < prob:
        return img
    delta = np.random.uniform(low, high)

    img += delta
    img = np.array(img).astype(np.uint8)
    return img

# 图像水平翻转
def apply_hflip(img):
    img = cv2.flip(img, 1)
    return img

color_maps = [
    apply_hue,
    apply_saturation,
    apply_contrast,
    apply_brightness,
    apply_hflip
]

def gen_random_ind():
    seed = random.random()
    if seed < 1 / 5:
        return 0
    elif seed >= 1 / 5 and seed < 2 / 5:
        return 1
    elif seed >= 2 / 5 and seed < 3 / 5:
        return 2
    elif seed >= 3 / 5 and seed < 4 / 5:
        return 3
    else:
        return 4

# 定义数据类型
class MyDataSet(Dataset):
    """
      步骤一：继承 paddle.io.Dataset 类
      """

    # def __init__(self, data_dir, label_path, transform=None):
    def __init__(self, data_paths, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(MyDataSet, self).__init__()
        self.data_list = []
        self.data = []
        for data_dir, label_path in data_paths:
            # data_dir = os.path.join(data_dir, label_path)
            # self.data_list.append([data_dir, label_path])
            label_path = os.path.join(data_dir, label_path)

            with open(label_path, encoding='utf-8') as f:

                data_set = json.loads(f.read())
                for data in data_set:
                    # print(data)
                    image_name = data["img_path"]
                    label = data["state"]
                    image_path = os.path.join(data_dir, image_name)
                    self.data_list.append([image_path, label])
        # print(self.data_list)
        # 传入定义好的数据处理方法，作为自定义数据集类的一个属性
        self.transform = transform
        self.flag_load_all = False

    # 一次读入所有数据到内存
    def load_alldata(self):
        if not self.flag_load_all:
            for image_path, label in self.data_list:
                img = Image.open(image_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                image = img.resize((128, 128), Image.Resampling.LANCZOS)
                image = np.array(image).astype(np.float32)
                self.data.append([image, label])
            self.flag_load_all = True

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        image = None
        label = None
        if self.flag_load_all:
            image, label = self.data[index]

        else:
            # 根据索引，从列表中取出一个图像
            image_path, label = self.data_list[index]
            # image = train_mapper(image_path)
            image = Image.open(image_path)
            # # cv2.imshow("temp", image)
            # # cv2.waitKey(0)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((128, 128), Image.Resampling.LANCZOS)

            image = np.array(image).astype(np.float32)


        # 随机图像增强
        id = gen_random_ind()
        image = color_maps[id](image)

        # 应用数据处理方法到图像上
        if self.transform is not None:
            image = self.transform(image)
        label = np.array(label[1:])
        if id==4:
            label = 0-label
        label = to_tensor(label, dtype="float32")
        # 返回图像和对应标签
        return image, label

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)


# In[14]:


def get_dataset():
    # 图像数据处理方法： ['BaseTransform', 'Compose', 'Resize', 'RandomResizedCrop', 'CenterCrop', 'RandomHorizontalFlip', 'RandomVerticalFlip', 'Transpose', 'Normalize', 'BrightnessTransform', 'SaturationTransform', 'ContrastTransform', 'HueTransform', 'ColorJitter', 'RandomCrop', 'Pad', 'RandomRotation', 'Grayscale', 'ToTensor', 'to_tensor', 'hflip', 'vflip', 'resize', 'pad', 'rotate', 'to_grayscale', 'crop', 'center_crop', 'adjust_brightness', 'adjust_contrast', 'adjust_hue', 'normalize']
    transform1 = Compose([Normalize(mean=[127.5], std=[127.5]),  ToTensor()])  #
    # path1 = ["data_set\image_set1201","data.json"]
    # path2 = ["data_set\image_set1206","data.json"]
    # path3 = ["data_set\image_set1208","data.json"]
    path3 = ["data_set\image_set","data.json"]
    data_paths = [path3]
    train_custom_dataset = MyDataSet(data_paths, transform=transform1)
    print("read all data to memory")
    train_custom_dataset.load_alldata()
    return train_custom_dataset


# In[15]:


from paddle import optimizer
from paddle.io import DataLoader
from paddle.vision.transforms import Resize, Compose, ColorJitter, Normalize, ToTensor
import paddle.nn.functional as F
import datetime, json
import csv
import os

def get_dataset(data_paths):
    # 图像数据处理方法： ['BaseTransform', 'Compose', 'Resize', 'RandomResizedCrop', 'CenterCrop', 'RandomHorizontalFlip', 'RandomVerticalFlip', 'Transpose', 'Normalize', 'BrightnessTransform', 'SaturationTransform', 'ContrastTransform', 'HueTransform', 'ColorJitter', 'RandomCrop', 'Pad', 'RandomRotation', 'Grayscale', 'ToTensor', 'to_tensor', 'hflip', 'vflip', 'resize', 'pad', 'rotate', 'to_grayscale', 'crop', 'center_crop', 'adjust_brightness', 'adjust_contrast', 'adjust_hue', 'normalize']
    transform4data = Compose([Normalize(mean=[127.5], std=[127.5]),  ToTensor()])  #
    train_custom_dataset = MyDataSet(data_paths, transform=transform4data)
    print("read all data to memory")
    train_custom_dataset.load_alldata()
    return train_custom_dataset

def train():
    # 按照 年月日_小时 时间格式定义模型保存路径
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H")
    model_parmas_path, model_opt_path, checkpoint_path = get_model_path(time_str)

    # 准备日志文件路径
    log_dir = "./data"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "loss_log.csv")

    # 打开日志文件（写入模式，先写表头）
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'type', 'loss_mae', 'loss_mse'])  # loss_mse 仅在 eval 时有值
    # 定义模型
    cnn_model = CnnModel()
    # 获取数据，并把数据转为 paddle 数据格式
    train_paths, eval_paths = get_data_paths()
    train_custom_dataset = get_dataset(train_paths)
    eval_custom_dataset = get_dataset(eval_paths)
    print("read ok")
    # 定义数据加载器
    train_loader = DataLoader(train_custom_dataset, batch_size=256, shuffle=True, drop_last=False, num_workers=0)
    test_loader = DataLoader(eval_custom_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=0)
    print("--------------------------")


    # 定义保存最后一次训练的检查点
    final_checkpoint = dict()
    # 定义优化器
    learning_rate = optimizer.lr.PiecewiseDecay(boundaries=[100, 400], values=[0.001, 0.0001, 0.00001])
    # 模型训练的配置准备，准备损失函数，优化器和评价指标
    opt = optimizer.Adam(learning_rate=learning_rate, parameters=cnn_model.parameters())
    # 绝对误差 MAE
    loss_fn = nn.L1Loss()
    # 模型训练
    print("start...")
    epoch_num = train_cfg["epochs"]

    for epoch_id in range(epoch_num):
        start_time = datetime.datetime.now()
        print("epoch {}, start time:{}".format(epoch_id,datetime.datetime.now()))
        # 将模型及其所有子层设置为训练模式。这只会影响某些模块，如Dropout和BatchNorm。
        cnn_model.train()
        loss_sum = 0
        # print()
        count = 0
        for batch_id, data in enumerate(train_loader()):

            # 训练数据获取输入输出
            x_data, y_data = data
            predicts = cnn_model(x_data)
            # print(predicts.shape)
            # 计算损失 等价于 prepare 中loss的设置
            loss = loss_fn(predicts, label=y_data)
            # loss = F.l1_loss(predicts, y_data)

            print("\tbatch {}, loss mae is: {}".format( batch_id, float(loss)))
            # 反向传播
            loss.backward()
            # 更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()

            final_checkpoint["loss"] = loss
            # 用于计算平均损失
            loss_sum = loss_sum + float(loss)
            count += 1

        print("train cost time:{}, avg_mae_loss:{}\n".format(datetime.datetime.now() - start_time, loss_sum/count))

        # 记录训练平均损失到日志文件
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch_id, 'train', loss_sum/count, ''])

        # 10个epoch进行一次评估
        if epoch_id !=0 and epoch_id % 10 == 0:
            print("eval epoch {}, start time:{}".format(epoch_id,datetime.datetime.now()))
            cnn_model.eval()
            loss_mse_sum = 0
            loss_mae_sum = 0
            count = 0
            for batch_id, data in enumerate(test_loader()):
                x_data, y_data = data
                predicts = cnn_model(x_data)
                # 计算损失，绝对误差和平方误差
                loss_mae = F.l1_loss(predicts, y_data)
                loss_mse = F.mse_loss(predicts, y_data)
                print("\tbatch {}, loss mae is: {}, mse is: {}".format( batch_id, float(loss_mae), float(loss_mse)))
                loss_mae_sum = loss_mae_sum + float(loss_mae)
                loss_mse_sum = loss_mse_sum + float(loss_mse)
                count += 1
            print("eval cost time:{}, avg_mae_loss:{}, avg_mse_loss:{}\n".format(datetime.datetime.now() - start_time, loss_mae_sum/count, loss_mse_sum/count))

            # 记录评估损失到日志文件
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([epoch_id, 'eval', loss_mae_sum/count, loss_mse_sum/count])

            # 保存Layer参数
        paddle.save(cnn_model.state_dict(), model_parmas_path)
        # paddle.static.save_inference_model(cnn_model.state_dict, "model/cnn23/dymic/cnn_lane.")

    # 保存Layer参数
    paddle.save(cnn_model.state_dict(), model_parmas_path)
    # 保存优化器参数
    paddle.save(opt.state_dict(), model_opt_path)
    # 保存检查点checkpoint信息
    paddle.save(final_checkpoint, checkpoint_path)

def save_jit():
    from paddle import jit
    cnn_model = CnnModel()
    # 如果保存模型用于推理部署，则需切换 eval()模式
    cnn_model.eval()
    last_name = get_last_model_name()
    infer_path = get_infer_path(last_name)
    # 载入模型参数、优化器参数和最后一个epoch保存的检查点
    model_params_path, _, _ = get_model_path(last_name)
    # 载入模型参数、优化器参数和最后一个epoch保存的检查点
    layer_state_dict = paddle.load(model_params_path)
    cnn_model.set_state_dict(layer_state_dict)

    layer = jit.to_static(cnn_model, input_spec=[paddle.static.InputSpec(shape=[None, 3, 128, 128], dtype='float32')]) # <----通过函数式调用 paddle.jit.to_static(layer) 一键实现动转静
    jit.save(layer, infer_path)

def infer_test():
    import time
    import numpy as np
    # imagenet上的归一化值
    transform1 = Compose([Resize((128,128)),Normalize(mean=[127.5], std=[127.5]), ToTensor("CHW")])  #    print("read data to memory")
    cnn_model = CnnModel()
    # 如果保存模型用于推理部署，则需切换 eval()模式
    cnn_model.eval()

    last_name = get_last_model_name()
    model_save_path, model_parmas_path, model_opt_path = get_model_path(last_name)
    # 载入模型参数、优化器参数和最后一个epoch保存的检查点
    layer_state_dict = paddle.load(model_save_path)

    # 将load后的参数与模型关联起来
    cnn_model.set_state_dict(layer_state_dict)

    train_custom_dataset = MyDataSet("train_data", "train.txt", transform=transform1)
    # 图像数据处理方法： ['BaseTransform', 'Compose', 'Resize', 'RandomResizedCrop', 'CenterCrop', 'RandomHorizontalFlip', 'RandomVerticalFlip', 'Transpose', 'Normalize', 'BrightnessTransform', 'SaturationTransform', 'ContrastTransform', 'HueTransform', 'ColorJitter', 'RandomCrop', 'Pad', 'RandomRotation', 'Grayscale', 'ToTensor', 'to_tensor', 'hflip', 'vflip', 'resize', 'pad', 'rotate', 'to_grayscale', 'crop', 'center_crop', 'adjust_brightness', 'adjust_contrast', 'adjust_hue', 'normalize']

    train_loader = DataLoader(train_custom_dataset, batch_size=16, shuffle=True, drop_last=False)

    cnn_model.train()
    for batch_id, data in enumerate(train_loader()):
            x_data, y_data = data
            predicts = cnn_model(x_data)
            print(predicts.numpy()[0][0])
            print(y_data.numpy()[0][0])
            time.sleep(3)

    print("start...")


# In[16]:


# 训练
train()


# In[ ]:





# In[17]:


# 保存为推理模型
save_jit()


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions.
