# smart_farm_models

第 21 届全国大学生智能车竞赛“智慧农场”项目的模型工程。仓库里放的是数据格式约定、数据检查、数据准备、离线增强、训练入口、模型导出和离线验收工具，以及两份可以直接下载使用的公开模型包。

摄像头采集、目标检测结果后处理、任务状态机、巡航输出接入、底盘控制、舵机控制、串口协议和 Jetson 上位机程序不在本仓库内。它们和本仓库的模型文件有接口关系，但不是同一个工程。模型在电脑上导出成功，也不等于上位机已经加载成功；上位机加载成功，也不等于整车在赛道上已经通过验收。

## 这份 README 解决什么问题

从一个干净的 Windows 工作目录开始，按本文可以完成以下工作：

- 安装并检查本项目使用的 Python、PaddlePaddle、PaddleX、Labelme 和 VisualDL 环境；
- 准备 00 车道分割、01 目标检测和 08 巡航回归所需的目录和数据文件；
- 运行仓库内所有可执行 Python 入口的 `--help`、静态检查和单元测试；
- 生成目标检测 COCO 数据集、按训练集做离线增强、训练 PP-YOLOE+、导出全 NMS 模型；
- 训练巡航回归模型、恢复断点、导出静态 Paddle 模型并做连续帧和光照压力评估；
- 解压仓库内的公开模型包，检查文件完整性，再把图片推理结果交给上位机联调。

有两件事必须先说清楚：

1. Git 仓库不保存原始图片、完整 checkpoint 和训练缓存。它们体积大，而且很多来源数据受原始采集环境和第三方工具授权约束。克隆仓库本身可以跑通代码检查，但要重做训练，必须另外准备数据和目标检测依赖。
2. 目标检测训练依赖仓外的 PaddleDetection 源码，Jetson 的 TensorRT、驱动、摄像头和上位机也不在这里。本文把版本边界、目录契约和命令写明，但不会把没有发生过的硬件实测写成电脑上的测试结果。

## 当前保留的三条主线

| 编号 | 目录 | 负责内容 | 当前状态 |
| --- | --- | --- | --- |
| 00 | `datasets/00_lane_seg`、`scripts/lane_seg` | Labelme 车道分割标注检查、划分清单、掩码导出 | 数据工具保留；仓库没有独立的正式深度学习训练入口 |
| 01 | `datasets/01_target_det`、`scripts/detection`、`scripts/onboard` | COCO 数据准备、检查、离线增强、PP-YOLOE+ 训练、导出、图片推理 | 正式目标检测主线 |
| 08 | `datasets/08_nav_control_optional`、`scripts/nav_control_optional` | 巡航记录清理、划分、回归训练、导出、连续帧和光照评估 | 可训练、可导出；不替代整车闭环 |

## 先看结论，再开始操作

本项目的复现顺序固定为：

1. 准备 Windows、Python、GPU 驱动和虚拟环境。
2. 运行通用布局检查和单元测试，确认代码环境正常。
3. 准备某一条主线的数据，只使用同一份类别表或同一套 session 划分。
4. 先做数据静态检查，再做离线增强；增强后再检查一次。
5. 训练前使用 dry-run 或巡航单批检查，确认输入和输出路径。
6. 导出明确的 checkpoint，计算 SHA256，保存训练配置和数据划分信息。
7. 先在电脑上做图片、连续帧和压力变体推理，再到目标 Jetson 上加载。
8. 用相同控制参数做实车 A/B，最后才决定替换上位机正在使用的模型。

不要用“loss 降了”“导出目录生成了”或“电脑上有检测框”替代最后三步。

## 从空目录开始

如果你还没有本地副本，先克隆公开仓库：

~~~powershell
Set-Location "E:\path\to\parent"
git clone https://github.com/indulgeuuu-del/smart_farm_models.git
Set-Location .\smart_farm_models
git log --oneline -1
~~~

公开仓库的 Git 工作树只包含代码、配置、数据元信息和两份公开模型 ZIP。原始图片、训练缓存和完整 checkpoint 在公开的 `data-archives-20260824` Release 中按分卷归档，而不在 Git 工作树中；外部 PaddleDetection 仍需按锁文件下载。`RELEASE_MANIFEST.json` 记录了归档名称、分卷规则和恢复前必须核对的文件。需要重做训练时，先恢复自己有权使用的原始数据或已发布的归档，再按本文对应主线的目录契约整理；不要因为 `datasets/` 目录存在就假定图片已经齐全。

从公开模型开始做联调时，不需要恢复训练数据；只需下载仓库内 ZIP、校验 SHA256、解压并运行“公开模型”章节的导出物检查。要运行图片推理，仍需要外部 PaddleDetection 的部署 `infer.py`。

## 目录总览

```text
configs/
  lane_seg.yaml
  target_det.yaml                                  通用占位配置，不能直接当 26/27 类配置
  target_det_fusion_20260731_26cls_*.yaml          26 类目标检测配置
  target_det_fusion_20260814_27cls_*.yaml          27 类目标检测配置
  nav_control_balanced_v2.yaml                     巡航 Balanced V2 参数契约
  nav_control_balanced_v3_light_corner.yaml       巡航 Balanced V3 参数契约

datasets/
  00_lane_seg/                                     车道分割元数据和示例结构
  01_target_det/                                   COCO/PaddleX 标注和数据报告
  08_nav_control_optional/                         巡航记录、划分和元数据

deploy/public_models/                              公开模型 ZIP 和 SHA256
env/                                               环境辅助脚本
scripts/common/                                    通用布局检查
scripts/lane_seg/                                  00 工具
scripts/detection/                                 01 数据、训练、导出和评估工具
scripts/onboard/                                   01 离线推理和导出物检查
scripts/nav_control_optional/                     08 工具
scripts/release/                                   公开数据归档生成和校验脚本
tests/                                             标准库 unittest 测试

assets/legacy/                                     历史截图和示意资料，不是训练输入
configs/                                           参数文件
LICENSE                                            MIT License
README.md                                          唯一的项目说明文档
```

`outputs/`、`tmp/`、`logs/`、`deploy/exported_models/` 和 `deploy/onboard/**/runs/` 是本地运行目录，默认不进入 Git。`datasets` 里的 JSON、JSONL、CSV、TXT 和 YAML 元数据可以进入 Git，图片文件不会随代码提交。公开的可直接使用模型以 `deploy/public_models/*.zip` 形式保存。

## Windows 环境准备

以下命令使用 PowerShell。PowerShell 5.1 和 PowerShell 7 均可；本文更推荐 PowerShell 7。所有命令都在仓库根目录执行。后续示例使用显式解释器路径，即使忘记激活虚拟环境也不会误调用系统 Python。

### 1. 检查系统前置条件

先安装：

- Git；
- Python 3.10，推荐 3.10.5；
- NVIDIA 显卡驱动；
- Windows x64；
- 需要从源码编译某些依赖时，安装 Visual C++ Build Tools。

打开新的 PowerShell，执行：

~~~powershell
git --version
py -0p
python --version
where.exe git
where.exe python
nvidia-smi
~~~

最低检查标准：

- `py -0p` 能找到 Python 3.10；
- `python --version` 不要是 Python 2 或者项目不支持的 3.12/3.13；
- `nvidia-smi` 能显示显卡、驱动版本和显存；
- `where.exe python` 的第一项不是一个意外的 Anaconda 或 Microsoft Store 路径。

如果机器没有 NVIDIA GPU，可以安装 CPU 分支并运行静态检查、数据工具和 CPU 推理；训练速度和 GPU 结果不能与参考环境比较。驱动版本必须不低于所安装 CUDA runtime 的要求，Paddle 的 CUDA wheel、驱动和显卡架构要一起选择，不能只看 Python 版本。

### 2. 进入仓库并处理 PowerShell 执行策略

把下面的路径换成实际克隆位置：

~~~powershell
Set-Location "E:\path\to\smart_farm_models"
$Repo = (Resolve-Path -LiteralPath .).Path
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
~~~

如果公司策略禁止修改执行策略，不要强行改系统策略。可以只在当前窗口临时放行：

~~~powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
~~~

确认位置正确：

~~~powershell
Get-Location
Test-Path -LiteralPath (Join-Path $Repo "README.md")
Test-Path -LiteralPath (Join-Path $Repo "scripts")
~~~

### 3. 创建主环境 `.venv`

先删除旧环境不是必需步骤。只有环境损坏、Python 小版本不一致或 `pip check` 无法修复时，才删除并重建；不要为了“清理”删除正在使用的环境。

~~~powershell
py -3.10 -m venv .venv
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
& $Py --version
& $Py -m pip install --upgrade pip setuptools wheel
~~~

若系统没有 `py` 启动器，改用：

~~~powershell
python -m venv .venv
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
~~~

激活是可选的。需要激活时：

~~~powershell
& (Join-Path $Repo ".venv\Scripts\Activate.ps1")
python --version
~~~

不激活时，始终使用 `$Py`。推荐后者，因为同一台电脑还要维护 Paddle 2.6.1 导出环境和 PaddleLite 环境。

### 4. 安装主环境的基础依赖

先安装不会决定 Paddle 后端的基础包。PaddlePaddle 要在下一步先选定 GPU 或 CPU 分支；PaddleX、Labelme 和 VisualDL 放到 Paddle 分支确认后再安装，避免 pip 在环境尚未确定时自行拉入不合适的 Paddle wheel：

~~~powershell
& $Py -m pip install `
  numpy==1.26.4 `
  pillow==12.1.0 `
  pyyaml==6.0.2 `
  opencv-python==4.5.5.64 `
  pandas==2.3.3
~~~

这里的版本是一次已经验证过的 Windows 参考环境，不表示所有 CUDA、驱动和 Python 组合都能使用同一组 wheel。若某个版本没有当前平台的 wheel，先确认 Python 是 3.10 x64，再根据该包官方的兼容矩阵选择同一主版本，不要把 32 位 Python 和 64 位 GPU wheel 混装。

安装完成后立即检查解释器和包来源：

~~~powershell
& $Py -m pip --version
& $Py -m pip show numpy pillow PyYAML opencv-python pandas
& $Py -m pip check
~~~

`pip check` 必须没有 dependency conflict。若同时出现 `opencv-python` 和 `opencv-contrib-python`，只保留项目实际需要的一个版本；二者同时存在时，`cv2` 到底加载哪一份不能靠猜。可以先查看：

~~~powershell
& $Py -c "import cv2; print(cv2.__version__); print(cv2.__file__)"
~~~

### 5. 安装 PaddlePaddle GPU 或 CPU 分支

先选择分支，不要两个分支都安装。

#### GPU 分支，参考环境

本项目已经验证过的一套主环境为：

| 项目 | 版本或说明 |
| --- | --- |
| Python | 3.10.5 |
| PaddlePaddle | `paddlepaddle-gpu==3.2.2` |
| CUDA runtime | 12.9 |
| NumPy | 1.26.4 |
| PaddleX | 3.4.3 |
| Labelme | 6.0.0 |
| VisualDL | 2.5.3 |
| OpenCV | 4.5.5.64 |
| GPU 参考 | NVIDIA RTX 4060 Laptop |

在能直接取得对应 wheel 的机器上执行：

~~~powershell
& $Py -m pip install paddlepaddle-gpu==3.2.2
~~~

如果默认 PyPI 没有当前 CUDA/平台的 wheel，打开 PaddlePaddle 官方安装页面，选择 Windows、Python 3.10、GPU、对应 CUDA 版本，复制官方给出的完整安装命令执行。不要把 CUDA 11.x 的命令复制到 CUDA 12.x 环境。安装后验证：

~~~powershell
& $Py -c "import paddle; print('paddle=', paddle.__version__); print('compiled_with_cuda=', paddle.is_compiled_with_cuda()); print('device=', paddle.device.get_device()); print('cuda_count=', paddle.device.cuda.device_count())"
~~~

预期结果是 `compiled_with_cuda=True`，设备类似 `gpu:0`，CUDA 设备数量大于 0。若 `nvidia-smi` 正常而 Paddle 显示 CPU，优先检查 wheel、Python 位数、驱动和 PATH，不要在脚本里绕过设备检查。

#### CPU 分支

先卸载 GPU wheel，再安装 CPU wheel：

~~~powershell
& $Py -m pip uninstall -y paddlepaddle-gpu
& $Py -m pip install paddlepaddle==3.2.2
& $Py -c "import paddle; print(paddle.__version__); print(paddle.is_compiled_with_cuda()); print(paddle.device.get_device())"
~~~

CPU 分支应该显示 `cpu`。不要用 CPU 结果判断 TensorRT FP16 是否可用。

### 6. 安装主环境工具包

确认 Paddle 分支已经正确后，再安装 PaddleX、Labelme 和 VisualDL：

~~~powershell
& $Py -m pip install `
  paddlex==3.4.3 `
  labelme==6.0.0 `
  visualdl==2.5.3
& $Py -m pip check
~~~

如果 pip 提示要替换 `paddlepaddle-gpu` 或 `paddlepaddle`，先停止安装并核对当前 PaddleX 版本的官方兼容说明。不要为了消除一个 resolver 提示而自动升级 Paddle 主版本；版本改变后要重新做设备检查和单元测试。

### 7. 安装和验证 VisualDL

目标检测训练脚本通过 `--use-vdl` 把训练曲线写入 VisualDL 日志目录。验证命令：

~~~powershell
& $Py -m visualdl --help
~~~

训练时使用：

~~~powershell
& $Py scripts\detection\train_target_det.py `
  --config .\configs\target_det_fusion_20260814_27cls_balanced_light_robust_v2_ppyoloe_plus_s_416_80e.yaml `
  --dataset-dir .\datasets\01_target_det\<your_augmented_dataset> `
  --output-dir .\outputs\01_target_det\run_name `
  --use-vdl `
  --no-amp
~~~

查看日志时使用 `visualdl --logdir <训练输出中的日志目录>`。VisualDL 只负责显示已写入的日志，不会修复训练失败或补齐缺失指标。

### 8. 单独创建 Paddle 2.6.1 导出环境

主训练环境和静态导出环境必须分开。当前参考导出环境为 Python 3.10.5、`paddlepaddle-gpu==2.6.1.post112`。2.6.1 的 wheel 必须选择与操作系统和 CUDA 匹配的版本；`post112` 这个后缀表示 CUDA 11.2 兼容构建，不能直接套到 CUDA 12.x 的机器上。

创建环境：

~~~powershell
py -3.10 -m venv .venv_paddle261_export
$ExportPy = Join-Path $Repo ".venv_paddle261_export\Scripts\python.exe"
& $ExportPy -m pip install --upgrade pip setuptools wheel
& $ExportPy -m pip install numpy==1.26.4 pillow==12.1.0 pyyaml==6.0.2 opencv-python==4.5.5.64
& $ExportPy -m pip install paddlepaddle-gpu==2.6.1.post112
& $ExportPy -c "import paddle; print(paddle.__version__); print(paddle.device.get_device())"
& $ExportPy -m pip check
~~~

若官方没有提供当前 Windows/CUDA 组合的 `2.6.1.post112` wheel，不能用 `3.2.2` 冒充导出环境。要么在有官方 2.6.1 wheel 的兼容环境中导出，要么明确把导出版本记录为另一套版本，并重新在 Jetson 上验证。

### 9. 单独创建 PaddleLite 转换环境

PaddleLite 只用于把已经导出的静态模型转换为 `.nb`。它不用于目标检测训练，也不用于替代 Paddle 3.x 主环境。当前参考版本为 `paddlelite==2.13rc0`：

~~~powershell
py -3.10 -m venv .venv_paddlelite_opt
$LitePy = Join-Path $Repo ".venv_paddlelite_opt\Scripts\python.exe"
& $LitePy -m pip install --upgrade pip setuptools wheel
& $LitePy -m pip install paddlelite==2.13rc0
& $LitePy -m pip check
& $LitePy (Join-Path $Repo ".venv_paddlelite_opt\Scripts\paddle_lite_opt") --help
~~~

不同 PaddleLite wheel 的命令入口可能显示为 `paddle_lite_opt`。Windows 下对无扩展名的包装脚本，使用当前环境的 Python 显式调用：

~~~powershell
& $LitePy -m pip show paddlelite
Get-ChildItem (Join-Path $Repo ".venv_paddlelite_opt\Scripts") | Where-Object Name -Match "lite|opt"
& $LitePy (Join-Path $Repo ".venv_paddlelite_opt\Scripts\paddle_lite_opt") --help
~~~

PaddleLite 转换前要确认静态模型的输入尺寸、输入名称、输出顺序和目标设备支持的算子；转换成功只说明转换器接受了模型，不说明上位机的预处理和后处理已经匹配。

### 10. 三套环境的使用边界

| 环境 | 用途 | 不要用来做什么 |
| --- | --- | --- |
| `.venv` | 数据检查、00 工具、01 数据工具和训练、08 训练及离线评估 | 不要加载只适配 Paddle 2.6.1 的旧导出链 |
| `.venv_paddle261_export` | 需要 Paddle 2.6.1 兼容范围的静态导出和复核 | 不要和主环境的训练 checkpoint 混用后声称结果等价 |
| `.venv_paddlelite_opt` | 静态模型到 `.nb` 的 PaddleLite 转换 | 不要用来训练、评估或运行 PaddleDetection |

每次切换环境都执行一次：

~~~powershell
& $Py -c "import sys; print(sys.executable)"
& $ExportPy -c "import sys; print(sys.executable)"
& $LitePy -c "import sys; print(sys.executable)"
~~~

看到的路径必须分别指向三个目录。不要只看命令提示符前面的环境名称。

## 第一次安装后的统一自检

在仓库根目录执行：

~~~powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
& $Py -c "import sys, numpy, PIL, yaml, cv2, paddle, paddlex, labelme, visualdl; print(sys.version); print('numpy', numpy.__version__); print('pillow', PIL.__version__); print('pyyaml', yaml.__version__); print('opencv', cv2.__version__); print('paddle', paddle.__version__); print('paddlex', paddlex.__version__); print('labelme', labelme.__version__); print('visualdl', visualdl.__version__); print('device', paddle.device.get_device())"
& $Py -m pip check
& $Py -m compileall -q scripts tests
~~~

如果其中一条命令失败，先修环境，不要继续训练。尤其注意：`PIL.__version__` 是 Pillow 版本；`cv2.__version__` 是实际加载的 OpenCV 版本；`paddle.device.get_device()` 是当前进程看到的设备，不是 `nvidia-smi` 的输出。

## 外部 PaddleDetection 环境

01 的训练、导出和图片推理调用 PaddleDetection 的 `tools/train.py`、`tools/export_model.py`、`ppdet` 和 `deploy/python/infer.py`，因此它是唯一必须在仓外取得的源码依赖。仓库用 `third_party/paddledetection.lock.json` 固定官方上游的 `v2.8.1` 和精确提交 `59d5f5ebebc2a380f5f07dd413a056b75af01f2a`；源码会克隆到被 Git 忽略的 `third_party/PaddleDetection/`，不会被当作本项目提交。

### 安装锁定源码

先安装主训练环境，再执行：

~~~powershell
.\env\bootstrap_environment.ps1 -Profile training
.\env\install_paddledetection.ps1

$PaddleDet = Join-Path $Repo "third_party\PaddleDetection"
$env:SMART_FARM_PADDLEDET_ROOT = (Resolve-Path -LiteralPath $PaddleDet).Path
git -C $PaddleDet rev-parse HEAD
Test-Path (Join-Path $PaddleDet "tools\train.py")
Test-Path (Join-Path $PaddleDet "tools\export_model.py")
Test-Path (Join-Path $PaddleDet "ppdet\__init__.py")
Test-Path (Join-Path $PaddleDet "deploy\python\infer.py")
& $Py -c "import sys; sys.path.insert(0, r'$PaddleDet'); import ppdet; print(ppdet.__file__)"
~~~

四个 `Test-Path` 都必须为 `True`，`rev-parse` 输出必须与锁文件中的 40 位提交号相同。`install_paddledetection.ps1` 遇到脏的第三方目录会主动停止，避免覆盖本地修改。需要安装该版本的附加依赖时，用下面命令；它可能改变主环境的依赖版本，因此随后必须重新运行 `pip check` 和测试：

~~~powershell
.\env\install_paddledetection.ps1 -InstallDependencies
& $Py -m pip check
~~~

### 脚本如何寻找 PaddleDetection

01 脚本依次使用：

1. 命令中的 `--paddledet-root`；
2. `SMART_FARM_PADDLEDET_ROOT` 环境变量；
3. `third_party/PaddleDetection/`；
4. `.venv/src/` 下可识别的 PaddleDetection 目录。

推荐始终显式传入 `$PaddleDet`。目标检测模型文件的格式解析由仓库自己的 `scripts/common/target_det_model.py` 完成，支持 `model.pdmodel`、`model.json`、`inference.pdmodel` 和 `inference.json`，不需要修改外部 `infer.py`。

## 通用检查和脚本入口

### 运行全部可执行入口的帮助

仓库中除 `scripts/nav_control_optional/legacy/` 外的 Python 文件都属于当前代码或测试工具。下面的循环只读取帮助，不会训练、写数据或加载模型：

~~~powershell
$Scripts = Get-ChildItem .\scripts -Recurse -Filter *.py |
  Where-Object { $_.FullName -notmatch "[\\/]legacy[\\/]" }
foreach ($Script in $Scripts) {
  Write-Host "===== $($Script.FullName) ====="
  & $Py $Script.FullName --help
  if ($LASTEXITCODE -ne 0) { throw "--help failed: $($Script.FullName)" }
}
~~~

`scripts/nav_control_optional/legacy/Xunhang_Official.py` 是旧的 Notebook 导出脚本，文件中包含 `get_ipython()`、旧目录和旧模型定义。它不是当前复现主线，不能作为普通 Python 命令运行。当前主线对应的巡航训练入口是 `train_nav_control_reg.py`。

### 通用目录检查、编译和测试

~~~powershell
& $Py scripts\common\check_mandatory_data_layout.py --datasets-root .\datasets
& $Py -m compileall -q scripts tests
& $Py -m unittest discover -s tests -q
~~~

仓库测试使用 Python 标准库 `unittest`，默认不需要 PaddleDetection checkout。`test_paddledet_assigner_compat.py` 是可选的上游集成测试：只有在锁定源码已经安装、且明确设置 `SMART_FARM_RUN_PADDLEDET_INTEGRATION=1` 时才会运行。未设置时它会显示为 `skipped`，不是失败。

~~~powershell
# 默认仓库检查
& $Py -m unittest discover -s tests -v

# 可选：验证锁定 PaddleDetection 的 assigner 接口
$env:SMART_FARM_PADDLEDET_ROOT = (Resolve-Path .\third_party\PaddleDetection).Path
$env:SMART_FARM_RUN_PADDLEDET_INTEGRATION = "1"
& $Py -m unittest tests.test_paddledet_assigner_compat -v
Remove-Item Env:SMART_FARM_RUN_PADDLEDET_INTEGRATION
~~~

通用布局检查会检查三条主线当前要求的元数据目录。因为图片和训练输出没有进入 Git，刚克隆的仓库如果缺少本地原始图片，针对图片存在性的检查可能失败，这是数据未恢复，不是代码通过。

### 输出目录约定

本项目脚本不会把训练结果写回 `configs/` 或 `datasets/`：

| 内容 | 推荐目录 | 是否提交 Git |
| --- | --- | --- |
| 训练日志和 checkpoint | `outputs/<task>/<run_name>/` | 否 |
| 离线推理结果 | `deploy/onboard/**/runs/` 或 `tmp/` | 否 |
| 导出物临时目录 | `deploy/exported_models/` | 否，最终包另行整理 |
| 数据诊断报告 | `outputs/` | 否 |
| 本地进度记录 | `logs/` | 否 |
| 公开模型 | `deploy/public_models/` | 是，ZIP 和 SHA256 |

同一目录不要混放不同类别数、不同类别顺序或不同输入尺寸的模型。每次正式训练使用新的 `run_name`，把配置文件、数据集目录、外部源码提交号、Paddle 版本和随机种子一起记录。

## 公开数据归档的生成与验收

这部分只适用于你已经确认拥有原始图片、训练输出和运行记录的公开或再发布权限时。它不会从 Git 工作树凭空生成原始数据，也不会把被 `.gitignore` 忽略的文件自动变成可公开文件。脚本只扫描固定的历史数据和运行结果目录：`Dataset_all`、`MyDataset`、`datasets`、`outputs`、`output`、`bbox_pr_curve`、`vdl_log_dir`、`tmp`、`logs`、`deploy/exported_models` 和 `deploy/onboard_handoff`。

### 生成归档

Windows 10/11 通常自带 `tar.exe`。先确认它存在，并把输出目录放在系统临时目录或仓库之外；不要把输出目录放进 `datasets`、`tmp` 或其他输入目录。脚本会拒绝删除没有专用标记的已有目录，也会拒绝把输出目录放在输入根目录中：

~~~powershell
$Repo = (Resolve-Path .).Path
$ReleaseDir = Join-Path $env:TEMP "smart_farm_models_public_release"
Get-Command tar.exe
& .\scripts\release\build_public_data_release.ps1 `
  -Root $Repo `
  -WorkDir $ReleaseDir `
  -TargetGiB 1.55
~~~

脚本会按源文件字节数排序并分卷，真实生成 `part-001.tar.gz` 等归档，同时生成 `source_manifest.tsv`、`part_plan.csv`、`scope.json`、`SHA256SUMS.txt` 和 `RELEASE_NOTES.md`。`scope.json` 只写仓库名称和统计信息，不写本机绝对路径。为了不把主仓的进度记录带到公开资产中，`logs/` 下文件名包含 `progress` 的 Markdown 日志会被明确排除；新的归档不要用手工复制代替脚本。

### 生成后验收

先做不解压的完整校验。这个步骤会读取每个压缩包、校验 SHA256、检查 tar 成员路径安全、检查分卷的文件数和字节数，并确认所有清单文件都被归档覆盖：

~~~powershell
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
& $Py scripts\release\verify_public_release.py `
  --release-dir $ReleaseDir `
  --json
~~~

要做恢复演练，必须使用空目录；验证器会拒绝向已有文件的目录写入，并在逐个安全提取后重新比较路径和文件大小：

~~~powershell
$RestoreDir = Join-Path $env:TEMP "smart_farm_models_public_restore"
if (Test-Path -LiteralPath $RestoreDir) {
  Remove-Item -LiteralPath $RestoreDir -Recurse -Force
}
& $Py scripts\release\verify_public_release.py `
  --release-dir $ReleaseDir `
  --extract-dir $RestoreDir
~~~

只有两个命令都返回 `STATUS=OK`，才可以把 `part-*.tar.gz` 和五个配套文件上传到 Release。上传后应从 GitHub 重新下载一遍并重复校验；本地生成成功不等于远端附件没有损坏。公开 Release 的 tag、附件名称和 SHA256 清单一旦交付，后续变更应使用新 tag，不要静默替换旧资产。

## 车道分割（00）

00 保留的是车道分割数据工具，不是一个已经完成训练和部署的车道分割产品。`configs/lane_seg.yaml` 明确把正式车道识别限定为深度学习路线，禁止把 OpenCV 阈值、边缘、轮廓或形态学处理当作正式车道识别模型。

### 数据目录

准备完整数据时，目录必须是：

~~~text
datasets/00_lane_seg/
  raw/                              原始图片，jpg/jpeg/png/bmp
  labelme_json/                     与图片对应的 Labelme JSON
  paddlex/
    images/train/
    images/val/
    annotations/train/
    annotations/val/
    class_names.txt
    train.txt
    val.txt
  metadata/                         可选的数据说明
  splits/                           split_dataset.py 输出的清单
~~~

固定类别和编号：

~~~text
0 background
1 road
2 border
3 cross_zone
~~~

Labelme JSON 的要求来自 `check_lane_seg_dataset.py`：

- `imagePath` 只能指向 `raw/` 内的文件名，不能写绝对盘符；
- `imageWidth` 和 `imageHeight` 必须是正整数；
- 每个 JSON 至少有一个 shape；
- shape 只允许 `polygon` 或 `rectangle`；
- polygon 至少三个点，rectangle 恰好两个点；
- 标签只能是 `background`、`road`、`border`、`cross_zone`；
- 文件名只能包含 ASCII 字母、数字、下划线、点和短横线。

### 检查、划分和导出

先检查完整目录：

~~~powershell
& $Py scripts\lane_seg\check_lane_seg_dataset.py `
  --dataset-dir .\datasets\00_lane_seg
~~~

生成确定性的训练/验证 JSON 文件名清单：

~~~powershell
& $Py scripts\lane_seg\split_dataset.py `
  --dataset-dir .\datasets\00_lane_seg `
  --labelme-dir .\datasets\00_lane_seg\labelme_json `
  --output-dir .\datasets\00_lane_seg\splits `
  --val-ratio 0.2 `
  --seed 2026
~~~

这个脚本只写 `train.txt` 和 `val.txt`，不复制图片、不生成掩码、不训练。它按 JSON 文件名排序后用种子打乱，再写出两个清单；它不会按 session 自动隔离连续帧。

导出训练掩码：

~~~powershell
& $Py scripts\lane_seg\export_seg_masks.py `
  --raw-dir .\datasets\00_lane_seg\raw `
  --labelme-dir .\datasets\00_lane_seg\labelme_json `
  --output-dir .\datasets\00_lane_seg\paddlex `
  --split train
~~~

验证集重复执行一次，把 `--labelme-dir` 换成只包含验证标注的目录，把 `--split` 换成 `val`。导出脚本会遍历传入目录下的全部 JSON，不会读取 `splits/train.txt` 或 `splits/val.txt`，因此不能把同一个 Labelme 目录分别传给两个 split。

转换逻辑是单通道 PNG 掩码：背景为 0，其余类别使用固定编号；rectangle 会转成四点 polygon；同一像素被多个 shape 覆盖时，后出现的 shape 会覆盖前面的值。需要人工确认 shape 顺序和重叠区域是否符合标注意图。

Labelme 样例启动探针：

~~~powershell
& .\env\launch_labelme_lane_seg_sample.ps1 -ProbeOnly
~~~

这个 PowerShell 脚本要求本地存在样例图片和标签目录。它会打印 Python、Labelme 版本、图片目录、标签文件和输出目录；不带 `-ProbeOnly` 时会启动 Labelme，并把日志写到本地 `logs/labelme/`。公开仓库没有把原始样例图片提交进 Git，缺少样例时应先补齐数据目录。

当前 00 主线没有独立的 PaddleX 训练和导出脚本。完成掩码导出不代表车道分割模型已经训练，更不代表 08 巡航模型的输入已经生成。

## 目标检测（01）

01 使用 COCO 标注格式和 PaddleDetection 的 PP-YOLOE+ small 416 配置。目标检测正式训练要求使用“只增强 train、原样保留 val/test”的离线增强目录。训练脚本默认拒绝没有 `augmentation_meta.json` 且目录名不含 `aug` 的数据集；这个保护不能用来掩盖数据准备缺失。

### 可训练数据目录

一个独立的 PaddleX/COCO 训练目录必须是：

~~~text
<dataset>/
  images/
    <image files>
  annotations/
    instance_train.json
    instance_val.json
    instance_test.json
  class_names.txt
  label_list.txt                       可选但建议保留
  data.yml                             可选但建议保留
  augmentation_meta.json               正式增强目录必须有
~~~

三个 JSON 都是 COCO 根对象，至少包含 `images`、`annotations` 和 `categories`。`images[].file_name` 是相对于 `<dataset>/images` 的安全相对路径；不能有绝对盘符、`..` 或路径穿越。图片的 `width`、`height` 要和真实图片一致。bbox 使用 COCO 的 `[x, y, width, height]`，宽高必须为正且不能超出图片边界；annotation 的 `image_id` 和 `category_id` 必须能在对应列表中找到。

类别表的四个地方必须同步：

1. COCO `categories` 的 id/name 顺序；
2. `class_names.txt`；
3. 配置文件的 `num_classes`；
4. 上位机使用的 label order。

COCO 的 category id 通常从 1 开始，而模型输出的 class index 从 0 开始。不要把 category id 直接当成模型索引。

### 公开 27 类类别表

当前公开 27 类元数据的顺序如下。训练另一份数据时，以那份数据自己的 `class_names.txt` 为准，不要只复制这张表：

~~~text
0  water_l3
1  water_l2
2  water_l1
3  water
4  order
5  cylinder_set
6  cylinder_3
7  cylinder_2
8  cylinder_1
9  ball_yellow
10 ball_blue
11 animal
12 name
13 danyuan_2
14 danyuan_1
15 storage
16 lable_yellow
17 lable_blue
18 rape
19 broccoli
20 potato
21 celery
22 mushroom
23 flammulina velutipes
24 tomato
25 green bean
26 green pepper
~~~

### 从 EasyData/COCO 来源准备数据

单个带 `Images/`、`Annotations/` 和 `coco_info.json` 的来源，使用：

~~~powershell
& $Py scripts\detection\prepare_formal_target_det_dataset.py `
  --source-dir .\MyDataset\My_Formal_Target_0519 `
  --output-dir .\datasets\01_target_det\paddlex_formal_0519 `
  --val-ratio 0.2 `
  --seed 20260519 `
  --force
~~~

准备脚本会读取来源 COCO，复制被 COCO 引用的图片和标注，按固定随机种子划分 train/val，并记录没有出现在 COCO 中的图片。`--force` 会覆盖目标目录；运行前确认目标目录没有要保留的内容。

多个来源合并时，使用合并脚本，不要手工复制同名图片或手工重写 category id：

~~~powershell
& $Py scripts\detection\merge_target_det_coco_datasets.py `
  --source-dir .\MyDataset\My_Formal_Target_0619 `
  --source-dir .\MyDataset\Target_name `
  --source-dir .\MyDataset\Target_order `
  --source-dir .\MyDataset\Target_storage0621 `
  --output-dir .\datasets\01_target_det\fusion_example `
  --val-ratio 0.2 `
  --test-ratio 0.1 `
  --split-group-size 8 `
  --seed 20260824 `
  --force
~~~

`--source-dir` 可以是包含 `Images`/`Annotations` 的内层目录，也可以是外层数据集目录。合并脚本会为图片生成带来源前缀的唯一文件名，按来源分组划分，重新生成 COCO 标注和类别表。来源之间类别名不一致时，先统一标签契约；不要用 `--drop-empty-categories` 掩盖漏标。该选项只应该在确认某类别确实不属于当前任务时使用。

如果原始数据已经整理成 `images/` 和 `annotations/instance_*.json`，不要再次走 EasyData 导入，直接把它当作 `<dataset>` 进行检查和增强。

### 导入 Dataset_all

旧的总数据目录若采用以下结构：

~~~text
Dataset_all/
  Images/
  train.json
  valid.json
  test.json
  label_list.txt
  data.yml
~~~

可用导入脚本转成仓库标准目录：

~~~powershell
& $Py scripts\detection\export_target_det_coco.py `
  --dataset-all-dir .\Dataset_all `
  --dataset-dir .\datasets\01_target_det
~~~

导入前确认 `Dataset_all` 内图片都能被三个 JSON 正确引用。导入脚本会重建目标目录中的原始 COCO/PaddleX 结果；不要在目标目录中放需要保留的手工文件。导入完成后先检查，再进行任何增强。

### COCO 静态检查

检查由 `check_det_coco_annotations.py` 完成。`--all` 只扫描 `datasets/01_target_det/annotations_coco/`，图片根目录固定为 `datasets/01_target_det/raw`；它不会自动扫描任意训练目录：

~~~powershell
& $Py scripts\detection\check_det_coco_annotations.py `
  --all `
  --datasets-root .\datasets `
  --allow-custom-classes
~~~

检查实际训练目录时，必须显式传图片目录和每个 JSON：

~~~powershell
& $Py scripts\detection\check_det_coco_annotations.py `
  --task target_det `
  --image-dir .\datasets\01_target_det\fusion_example\images `
  --coco-json .\datasets\01_target_det\fusion_example\annotations\instance_train.json `
  --coco-json .\datasets\01_target_det\fusion_example\annotations\instance_val.json `
  --coco-json .\datasets\01_target_det\fusion_example\annotations\instance_test.json `
  --allow-custom-classes
~~~

`--allow-custom-classes` 只放宽类别表是否等于脚本内置旧类别，不会修复类别映射。以下问题仍然会使检查失败：JSON 无法解析、缺失图片、重复 id、重复类别名、空类别表、非法相对路径、负数或越界 bbox、annotation 引用了不存在的 image/category。

增强之后必须用同一条命令再检查一次，并把路径换成增强目录。不要用 `--all` 代替这一步，否则可能只查到未增强的数据。

### 人工抽检和数据报告

把 COCO 框画到图片上，并生成 HTML 索引、图片清单和疑似问题 CSV：

~~~powershell
& $Py scripts\detection\visualize_target_det_annotations.py `
  --dataset-dir .\datasets\01_target_det\fusion_example `
  --output-dir .\outputs\01_target_det\label_visual_review `
  --splits train val test
~~~

默认会标记：不同类别框的高 IoU 重叠、占图面积过小的框和占图面积过大的框。这些只是抽检线索，不是自动删图结论。机器标注数据必须逐图检查类别、框边界、漏标、重复框和空图片。

生成类别数量、框尺寸、弱类样本和 Markdown 报告：

~~~powershell
& $Py scripts\detection\analyze_target_det_dataset.py `
  --dataset-dir .\datasets\01_target_det\fusion_example `
  --output-dir .\outputs\01_target_det\diagnostics `
  --weak-threshold 100 `
  --sample-limit 12
~~~

`analyze_target_det_dataset.py` 不预设类别数量，也不把历史类别表当成当前数据的真值。它会分别读取每个 COCO JSON 的 `categories` 列表，并把列表顺序作为类别顺序；因此 15 类、26 类、27 类或其他合法类别数量都可以使用同一条命令。`train`、`val`、`test` 三个 split 的类别 ID、类别名称和顺序必须完全一致，否则脚本会直接报错。脚本还会拒绝空类别表、重复类别 ID、重复类别名、空类别名，以及 annotation 引用不存在的 `category_id`。

报告中的 `class_names`、`num_classes`、`class_counts` 和 `class_size_counts` 都来自当前 COCO 文件，不会自动补入没有出现在数据集里的历史类别。类别表仍然必须先通过 COCO 静态检查；该分析脚本负责统计和抽样，不替代图片逐张人工复核。

### 只作用于 train 的离线增强

离线增强脚本不会修改源目录。它会复制原始 train/val/test，给 train 追加变体，原样复制 val/test，并写入 `augmentation_meta.json`：

~~~powershell
$Source = ".\datasets\01_target_det\fusion_example"
$Aug = ".\datasets\01_target_det\fusion_example_aug"
& $Py scripts\detection\augment_target_det_coco.py `
  --source-dir $Source `
  --output-dir $Aug `
  --augmentations flip exposure blur noise shadow `
  --variants-per-image 1 `
  --seed 20260824 `
  --force
~~~

支持的增强名称：

~~~text
flip vflip affine perspective zoomout
exposure blur noise shadow
lowres overexposure underexposure mixed_lighting
object_flip object_vflip object_rotate90 object_rotate180 object_rotate270
~~~

脚本还可以对 hard-case、focus label 和 zoom-out label 额外生成变体：

~~~powershell
& $Py scripts\detection\augment_target_det_coco.py `
  --source-dir $Source `
  --output-dir $Aug `
  --augmentations flip exposure blur noise shadow lowres overexposure underexposure mixed_lighting `
  --variants-per-image 1 `
  --hard-case-labels cylinder_1 cylinder_2 cylinder_3 water_l1 water_l2 water_l3 name `
  --hard-case-extra 1 `
  --focus-labels name `
  --focus-extra 1 `
  --focus-augmentations zoomout lowres `
  --zoomout-labels name `
  --zoomout-extra 1 `
  --seed 20260824 `
  --force
~~~

先用小规模 `--max-augmented-train-images` 做 smoke test，确认输出文件和 bbox，再生成完整目录。`--force` 会覆盖目标目录。增强后必须检查 `augmentation_meta.json`，确认只增加 train 图片，val/test 的文件名、annotation 数量和内容没有被增强污染。

### 数据集分析与 hard-case 工具

下面这些工具都要求对应的数据或模型存在：

| 脚本 | 用途 | 关键输出 |
| --- | --- | --- |
| `analyze_target_det_dataset.py` | 按 COCO 类别契约生成类别和框尺寸报告 | JSON、Markdown、弱类样本 |
| `visualize_target_det_annotations.py` | 框可视化和疑似标注问题索引 | HTML、CSV、画框图片 |
| `evaluate_target_det_hardcases.py` | 对指定模型和 hard-case 图像做压力评估 | JSON/Markdown 评估报告 |
| `summarize_target_det_hardcases.py` | 汇总 hard-case 评估结果 | 汇总 JSON/Markdown |
| `build_nine_vegetable_capture_plan.py` | 生成九种蔬菜采集计划和 CSV | JSON、CSV |
| `monitor_target_det_training.py` | 读取训练日志和指标 | 监控报告 |
| `watch_export_target_det_best.py` | 监视 best checkpoint 并触发导出 | 导出目录和日志 |
| `expand_target_det_checkpoint.py` | 检查或扩展 checkpoint | 新 checkpoint 或检查结果 |

每个脚本的完整选项以当前版本的 `--help` 为准。脚本之间不会自动传递数据集路径；把一个脚本的输出目录显式传给下一个脚本。

### 27 类正式配置

当前公开的 27 类正式配置是：

~~~text
configs/target_det_fusion_20260814_27cls_balanced_light_robust_v2_ppyoloe_plus_s_416_80e.yaml
~~~

它的主要参数如下：

| 参数 | 值 |
| --- | --- |
| 模型 | PP-YOLOE+ small，CSPResNet + CustomCSPPAN |
| 预训练 | Obj365 PP-YOLOE CRN small 预训练权重 |
| 输入/导出尺寸 | 416 x 416 |
| epoch | 80 |
| 类别数 | 27 |
| 优化器 | Momentum，momentum 0.9 |
| base learning rate | 0.001 |
| 学习率 | CosineDecay，80 epoch，min lr ratio 0.02，末尾 plateau 4 epoch |
| warmup | LinearWarmup，5 epoch，start factor 0 |
| L2 | 0.0005 |
| 梯度裁剪 | norm 10.0 |
| EMA | 开启，decay 0.9998 |
| static assigner | ATSS，topk 9，切换 epoch 20 |
| dynamic assigner | TaskAligned，topk 13，alpha 1.0，beta 6.0 |
| loss 权重 | class 1.0，iou 2.5，dfl 0.5 |
| worker_num | 2 |
| train batch | 16，shuffle，drop_last |
| train 多尺度 | 352、384、416、448、480、512 |
| eval batch | 2，固定 416 |
| NMS | full NMS，top_k 1000，keep_top_k 300，score 0.01，NMS 0.7 |
| AMP | 正式口径为 FP32，命令中使用 `--no-amp` |

训练增强由配置中的 `RandomDistort`、`RandomExpand`、`RandomCrop`、`RandomFlip` 和 `BatchRandomResize` 组成；光照相关的更强变体来自上一步的 train-only 离线增强，不应加到 val/test。26 类正式配置的模型结构和部署口径相同，类别数、数据集和配置文件名不同；不要用 27 类权重加载 26 类类别表。

`configs/target_det.yaml` 只是默认占位配置，目前 `num_classes: 10`，不能直接用于当前 26 类或 27 类训练。配置里的 `save_dir` 和数据集目录是占位值，实际路径由训练脚本传入。

### 训练前的最小闭环

定义变量并先检查路径：

~~~powershell
$PaddleDet = $env:SMART_FARM_PADDLEDET_ROOT
if ([string]::IsNullOrWhiteSpace($PaddleDet)) { throw "SMART_FARM_PADDLEDET_ROOT is not set" }
if (-not (Test-Path (Join-Path $PaddleDet "tools\train.py"))) { throw "PaddleDetection tools/train.py not found" }

$Config = (Resolve-Path .\configs\target_det_fusion_20260814_27cls_balanced_light_robust_v2_ppyoloe_plus_s_416_80e.yaml).Path
$Dataset = (Resolve-Path .\datasets\01_target_det\fusion_example_aug).Path
$Run = (Join-Path $Repo "outputs\01_target_det\27cls_balanced_v2")
~~~

把 `fusion_example_aug` 换成实际增强目录。不要让 `$Dataset` 指向源目录或未增强目录。执行两次静态检查：

~~~powershell
& $Py scripts\detection\check_det_coco_annotations.py `
  --task target_det `
  --image-dir (Join-Path $Dataset "images") `
  --coco-json (Join-Path $Dataset "annotations\instance_train.json") `
  --coco-json (Join-Path $Dataset "annotations\instance_val.json") `
  --coco-json (Join-Path $Dataset "annotations\instance_test.json") `
  --allow-custom-classes
~~~

然后只打印将要执行的 PaddleDetection 命令：

~~~powershell
& $Py scripts\detection\train_target_det.py `
  --paddledet-root $PaddleDet `
  --config $Config `
  --dataset-dir $Dataset `
  --output-dir $Run `
  --no-amp `
  --dry-run
~~~

dry-run 不会生成 checkpoint。检查打印结果中是否包含正确的 config、数据目录、`save_dir` 和 FP32 开关；确认无误后才正式启动。

### 正式训练、VisualDL 和断点

~~~powershell
& $Py scripts\detection\train_target_det.py `
  --paddledet-root $PaddleDet `
  --config $Config `
  --dataset-dir $Dataset `
  --output-dir $Run `
  --use-vdl `
  --no-amp
~~~

PaddleDetection 的实际日志和 checkpoint 会落在 `$Run`。训练中断后，先查看该目录确认最后保存的 checkpoint，再使用与当前配置和数据一致的 `--resume` 参数续训。不要把不同类别数、不同类别顺序、不同输入尺寸或不同增强策略的旧 checkpoint 接到新实验上。新实验不要把旧 run 目录当作 `--output-dir` 复用。

训练日志中常见的 TensorRT 或算子兼容信息只说明 PaddleDetection 正在构建或跳过某个子图，不等于训练失败。真正失败要看进程退出码、最后一条异常和输出目录是否有完整 checkpoint。

### 导出全 NMS 模型

PaddleDetection 训练产生的 `best_model` 或 `model_final` 路径以实际输出为准。先找文件：

~~~powershell
Get-ChildItem $Run -Recurse -File |
  Where-Object { $_.Name -match "best|model_final|pdparams" } |
  Select-Object FullName, Length
~~~

正式模型导出示例：

~~~powershell
$Weights = Join-Path $Run "best_model"
$Export = Join-Path $Repo "deploy\exported_models\01_target_det\27cls_balanced_v2"

& $Py scripts\detection\export_target_det_model.py `
  --paddledet-root $PaddleDet `
  --config $Config `
  --weights $Weights `
  --dataset-dir $Dataset `
  --output-dir $Export `
  --trt `
  --use-gpu
~~~

如果实际最佳权重目录叫 `model_final`，只替换 `$Weights`，不要修改配置中的类别数。`--trt` 是导出时的 TensorRT 兼容开关，不代表电脑上已经完成 TensorRT FP16 实测；FP16 运行模式要在目标设备上另行验证。

导出目录至少要能找到以下一组文件：

~~~text
<export>/
  model.pdmodel 或 inference.pdmodel
  model.pdiparams 或 inference.pdiparams
  infer_cfg.yml
  label_list.txt 或 infer_cfg.yml 内的 label_list
~~~

### 导出物静态兼容检查

检查器不再内置历史类别表。它验证模型文件成对存在、`infer_cfg.yml` 可解析、架构为 YOLO、metric 为 COCO、Resize 尺寸为正、阈值有效、标签非空且无重复；若模型目录带 `label_list.txt`，还会强制它与 `infer_cfg.yml` 的标签顺序完全一致。用于上位机前，提供一份明确的类别表可以把“看起来没问题”变成严格顺序检查：

~~~powershell
& $Py scripts\onboard\check_target_det_baseline_compat.py `
  --model-dir $Export `
  --expected-labels-file (Join-Path $Dataset "class_names.txt") `
  --json
~~~

外部类别表、`label_list.txt` 和 `infer_cfg.yml` 的每一行必须完全一致；26 类和 27 类模型不能互换。

### 电脑图片推理

`run_target_det_onboard.py` 会加载锁定 PaddleDetection checkout 中的 `deploy/python/infer.py`。模型目录是必填项；推理前会先检查静态模型文件是否存在。运行前确认：

~~~powershell
$PaddleDet = Join-Path $Repo "third_party\PaddleDetection"
Test-Path -LiteralPath (Join-Path $PaddleDet "deploy\python\infer.py")
~~~

Paddle 图片推理：

~~~powershell
& $Py scripts\onboard\run_target_det_onboard.py `
  --image-dir (Join-Path $Dataset "images") `
  --model-dir $Export `
  --paddledet-root $PaddleDet `
  --output-dir .\deploy\onboard\01_target_det\runs\27cls_paddle `
  --device GPU `
  --run-mode paddle `
  --threshold 0.3
~~~

固定 416 输入的 TensorRT FP16 图片推理：

~~~powershell
& $Py scripts\onboard\run_target_det_onboard.py `
  --image-dir (Join-Path $Dataset "images") `
  --model-dir $Export `
  --paddledet-root $PaddleDet `
  --output-dir .\deploy\onboard\01_target_det\runs\27cls_trt_fp16 `
  --device GPU `
  --run-mode trt_fp16 `
  --trt-min-shape 416 `
  --trt-opt-shape 416 `
  --trt-max-shape 416 `
  --threshold 0.3
~~~

测试单张图片时，把 `--image-dir` 换成 `--image-file <path>`；二者不能同时传。`--max-images` 可用于先跑少量样本，`--no-save-images` 和 `--no-save-coco` 可减少磁盘输出。默认结果目录包含 `onboard_result.json`、`summary.json`，通常还会有画框图片和 `bbox.json`。summary 中的 FPS 是当前电脑、当前 batch 和当前 backend 的测量值，不是 Jetson 的承诺。

### 上车前必须检查的差异

电脑离线推理和 Jetson 上车运行至少有以下差异：

- Paddle、TensorRT、CUDA、cuDNN 和驱动版本不同；
- 上位机的预处理颜色顺序、Resize、归一化和阈值可能不同；
- 动态 shape 的 min/opt/max 配置可能不同；
- 上位机线程时序、摄像头帧率和显示路径可能影响结果；
- 模型文件能够被 Paddle 加载，不表示 PaddleLite `.nb` 或 TensorRT FP16 能加载。

因此，正式交付要把模型目录、类别表、输入尺寸、阈值、运行模式、设备版本和输出日志一起交给上位机队友。

`scripts/onboard/eval_onboard_samples.py` 用于没有标注的板端样本：它会检查模型结构、列出图片并写入 `sample_preflight.json`。默认不加载 PaddleDetection，也不会伪造 AP 或准确率；增加 `--run-inference` 才会调用真实图片推理：

~~~powershell
& $Py scripts\onboard\eval_onboard_samples.py `
  --samples-dir .\samples\onboard `
  --model-dir $Export `
  --output-dir .\deploy\onboard\01_target_det\sample_preflight\local_check

& $Py scripts\onboard\eval_onboard_samples.py `
  --samples-dir .\samples\onboard `
  --model-dir $Export `
  --paddledet-root $PaddleDet `
  --run-inference
~~~

## 巡航控制（08）

08 是当前仓库的巡航优先路线。模型输入是前视摄像头图像，网络输出形状固定为 `[batch, 2]`：`output_0` 是偏差回归值，`output_1` 始终为 0，只为兼容已有上位机接口。速度字段不会作为第二个模型输出参与训练；如果采集记录有 `state`，训练脚本取 `state[2]` 作为偏差，忽略 `state[0]` 和 `state[1]`。

### 原始记录格式

每个 source 目录包含图片和一份 `data.json` 或 `data.jsonl`：

~~~text
<source>/
  data.json 或 data.jsonl
  000000.jpg
  000001.jpg
  ...
~~~

记录可以是：

~~~json
{"img_path":"000000.jpg","state":[0.25,0.0,-0.12]}
~~~

也可以是：

~~~json
{"img_path":"000000.jpg","deviation":-0.12}
~~~

要求：

- `img_path` 是 source 内相对路径，只允许正向斜杠，不允许绝对路径、盘符、`.` 或 `..`；
- 图片必须真实存在，读取后转 RGB，resize 到 128 x 128；
- 输入归一化为 `(pixel - 127.5) / 127.5`，再转成 CHW；
- `state` 必须正好有三个有限数值，训练标签取第三项；
- `deviation` 必须是有限数值；
- `session` 如果存在必须是整数；
- `data.jsonl` 每行一个 JSON 对象，空行会跳过；
- `valid_for_training` 只由 `prepare_gamepad_nav_dataset.py` 过滤，普通训练脚本不会自动过滤。

### 目录和记录检查

检查仓库内已经整理的 source：

~~~powershell
& $Py scripts\nav_control_optional\check_nav_control_data.py `
  --dataset-dir .\datasets\08_nav_control_optional
~~~

如果只想检查两个明确的 train/val source：

~~~powershell
& $Py scripts\nav_control_optional\check_nav_control_data.py `
  --record-dir .\datasets\08_nav_control_optional\raw\train_source `
  --record-dir .\datasets\08_nav_control_optional\raw\val_source `
  --required-splits train,val
~~~

`check_nav_control_data.py` 会检查记录能否解析、图片是否存在、标签是否有限、范围统计和可选 split manifest。它不检查摄像头是否真的拍到了有效赛道，也不判断标签是否适合比赛。

### 生成 train/val source

已有一个连续 source 时，按原始顺序切分：

~~~powershell
& $Py scripts\nav_control_optional\prepare_nav_control_dataset.py `
  --source-dir .\datasets\08_nav_control_optional\raw\image_set1208 `
  --dataset-dir .\datasets\08_nav_control_optional `
  --train-ratio 0.7 `
  --val-ratio 0.15
~~~

训练和验证来源已经分别准备好时，显式指定两个目录：

~~~powershell
& $Py scripts\nav_control_optional\prepare_nav_control_dataset.py `
  --train-source-dir .\datasets\08_nav_control_optional\raw\image_set_r `
  --val-source-dir .\datasets\08_nav_control_optional\raw\image_set1208 `
  --dataset-dir .\datasets\08_nav_control_optional
~~~

正式巡航数据推荐按完整 session 隔离，而不是把相邻帧随机打散。相邻帧同时出现在 train 和 val 会让验证误差看起来很好，却不能说明模型能适应新弯道或新光照。

### 从手柄采集数据清理并按 session 划分

`prepare_gamepad_nav_dataset.py` 只读取 `valid_for_training: true` 的记录，并且不改写原始 source。它支持连续 block 划分和 session 划分。以 lane_set7 的公开划分口径为例：删除 session 51，session 52 和 54 作为验证集，删除每个保留 session 尾部 100 帧，并删除 `command_speed` 为零的记录：

~~~powershell
$CruiseSource = (Resolve-Path .\datasets\08_nav_control_optional\raw\lane_set7_original_labels_session54_val_20260724).Path
$CruisePrepared = (Join-Path $Repo "datasets\08_nav_control_optional\paddle_custom\lane_set7_balanced_v3_val52_54")

& $Py scripts\nav_control_optional\prepare_gamepad_nav_dataset.py `
  --source-dir $CruiseSource `
  --output-dir $CruisePrepared `
  --split-mode sessions `
  --val-sessions 52,54 `
  --drop-sessions 51 `
  --drop-zero-speed `
  --drop-tail-frames 100 `
  --overwrite
~~~

准备结果为：

~~~text
<prepared>/
  train/
    data.jsonl
    images...
  val/
    data.jsonl
    images...
  split_meta.json
  source_session_meta.json
~~~

`split_meta.json` 是训练复现所需的关键证据，应核对 `drop_sessions`、`val_sessions`、`train_sessions`、`removed_by_reason`、`train_records` 和 `val_records`。如果使用 `--max-speed`，脚本会保留原速度到 `source_command_speed` 或 `source_state0`，同时把训练字段改成指定上限；没有这个参数时不会改速度。

这一步的“删除尾帧”只是生成新的训练 source，不会删除原始图片。紫色停车区是否参与训练，取决于这里的过滤结果和来源标签，不能只看目录名下结论。

### Balanced V2 参数

`configs/nav_control_balanced_v2.yaml` 是实验参数契约。对应默认训练参数：

| 参数 | 值 |
| --- | --- |
| 模型 | `PaddleLaneCnnModel`，128 x 128 |
| epoch | 100 |
| batch size | 128 |
| eval batch size | 256 |
| 优化器 | AdamW |
| weight decay | 0.0001 |
| learning rate | 0.0005 |
| scheduler | cosine warmup |
| warmup | 3 epoch，从 0.00005 开始 |
| eta min | 0.00001 |
| eval interval | 每 2 epoch，最后一轮必评估 |
| early stop | 至少 30 epoch；连续 10 次验证无改善停止 |
| min delta | 0.0001 |
| 偏差损失权重 | `moderate_abs_bins` |
| 训练增强 | `robust_v2` |
| seed | 20260418 |
| 输出 | `[deviation, 0]` |

`robust_v2` 在训练集上做亮度、曝光、gamma、对比度、饱和度、色相、阴影、轻微模糊/噪声和水平翻转；水平翻转时同步改变偏差符号。验证集不做随机增强。

### Balanced V3 light corner 参数

`configs/nav_control_balanced_v3_light_corner.yaml` 记录的是一次更偏向弯道和光照鲁棒性的实验：

| 参数 | 值 |
| --- | --- |
| epoch | 120 |
| batch/eval batch | 128 / 256 |
| 优化器 | AdamW |
| learning rate | 0.00035 |
| scheduler | cosine warmup，5 epoch，start 0.000035，eta min 0.000005 |
| eval interval | 2 |
| early stop | 至少 40 epoch；连续 15 次验证无改善停止 |
| min delta | 0.00005 |
| 偏差损失权重 | `corner_focus_v1` |
| 增强 | `light_corner_v3` |
| seed | 20260807 |

这个 YAML 不会被训练脚本自动读取，也不会因为只写 `--preset balanced_v3_light_corner` 就自动切换所有值。要复现 V3，必须把 epoch、学习率、数据 source、过滤规则、损失权重和增强 profile 都显式写进命令。

### 运行训练前向检查

整理好 `train/` 和 `val/` 后，先用单批检查：

~~~powershell
$CruiseDataset = (Resolve-Path .\datasets\08_nav_control_optional).Path
$TrainSource = (Resolve-Path $CruisePrepared\train).Path
$ValSource = (Resolve-Path $CruisePrepared\val).Path

& $Py scripts\nav_control_optional\train_nav_control_reg.py `
  --preset balanced_v3_light_corner `
  --dataset-dir $CruiseDataset `
  --train-sources $TrainSource `
  --eval-sources $ValSource `
  --output-dir .\tmp\nav_check `
  --epochs 1 `
  --batch-size 8 `
  --eval-batch-size 8 `
  --max-train-batches 1 `
  --max-eval-batches 1 `
  --device gpu:0 `
  --check-only `
  --split eval
~~~

`--check-only` 只加载一批图像并做一次前向，不创建训练 checkpoint。预期输出包含输入形状 `[8, 3, 128, 128]`、标签形状 `[8, 2]` 和预测形状 `[8, 2]`。如果要在 CPU 检查，把 `--device gpu:0` 改成 `--device cpu`。

### 正式训练和断点续训

Balanced V3 的一次完整命令如下：

~~~powershell
$RunRoot = Join-Path $Repo "outputs\08_nav_control_optional"
$RunName = "balanced_v3_light_corner_reproduction"

& $Py scripts\nav_control_optional\train_nav_control_reg.py `
  --preset balanced_v3_light_corner `
  --dataset-dir $CruiseDataset `
  --train-sources $TrainSource `
  --eval-sources $ValSource `
  --output-dir $RunRoot `
  --run-name $RunName `
  --epochs 120 `
  --batch-size 128 `
  --eval-batch-size 256 `
  --optimizer adamw `
  --weight-decay 0.0001 `
  --lr 0.00035 `
  --lr-scheduler cosine_warmup `
  --lr-t-max 120 `
  --warmup-epochs 5 `
  --warmup-start-lr 0.000035 `
  --lr-eta-min 0.000005 `
  --eval-interval 2 `
  --early-stop-min-epochs 40 `
  --early-stop-patience-evals 15 `
  --early-stop-min-delta 0.00005 `
  --deviation-loss-weight-mode corner_focus_v1 `
  --augmentation-profile light_corner_v3 `
  --seed 20260807 `
  --device gpu:0
~~~

训练目录结构通常是：

~~~text
outputs/08_nav_control_optional/<run_name>/dynamic/
  cnn_lane.pdparams       当前最后一轮参数
  cnn_lane.pdopt          当前优化器状态
  cnn_lane.pkl            断点和历史
  metrics.json            配置、样本数、每轮指标、best 信息
  best/                   验证 MAE 最优
  train_best/             训练 MAE 最优
  corner_best/            弯道指标最优
  balanced_best/          全局误差和弯道指标综合最优
  eval_snapshots/         定期评估快照
~~~

`best`、`corner_best` 和 `balanced_best` 的含义不同。不能只因为某个 epoch 的全局 MAE 最低，就说它在锐角弯或两连直角急弯最好。比较模型时要同时看全局 MAE、弯道 MAE、过渡段误差、欠转率、方向错误率、连续帧尖峰和光照变体结果。

断点续训必须指向包含 `cnn_lane.pdparams`、`cnn_lane.pdopt`、`cnn_lane.pkl` 和 `metrics.json` 的动态目录，并指定同一个 `--run-name`：

~~~powershell
& $Py scripts\nav_control_optional\train_nav_control_reg.py `
  --preset balanced_v3_light_corner `
  --dataset-dir $CruiseDataset `
  --train-sources $TrainSource `
  --eval-sources $ValSource `
  --output-dir $RunRoot `
  --run-name $RunName `
  --epochs 160 `
  --resume-dir (Join-Path $RunRoot "$RunName\dynamic") `
  --device gpu:0
~~~

目标 epoch 必须大于 checkpoint 已完成 epoch。改变数据 source、类别或核心训练参数后不要续训旧目录；这会让指标不可比较，脚本也可能主动拒绝恢复。`--init-params-path` 是从已有参数初始化新实验，不等于断点续训，不能和 `--resume-dir` 同时使用。

### 导出巡航静态模型

先选择要导出的动态参数。下面选择综合结果目录中的 `balanced_best`；如果实车对比指定了 `corner_best` 或某个 epoch，替换 `$Params` 即可：

~~~powershell
$Dynamic = Join-Path $RunRoot "$RunName\dynamic"
$Params = Join-Path $Dynamic "balanced_best\cnn_lane.pdparams"
$CruiseExport = Join-Path $Repo "deploy\exported_models\08_nav_control_optional\$RunName"

& $Py scripts\nav_control_optional\export_nav_control_reg.py `
  --params-path $Params `
  --export-dir $CruiseExport `
  --device gpu:0
~~~

导出目录通常包含：

~~~text
cnn_lane.pdmodel
cnn_lane.pdiparams
cnn_lane.pdiparams.info
export_meta.json
~~~

`.nb` 不是这个脚本直接生成的结果；需要在 PaddleLite 环境中另行转换。导出脚本会重新加载参数做检查，仍然要用下面的静态推理命令验证输入和输出形状。

### 动态模型检查

~~~powershell
& $Py scripts\nav_control_optional\check_nav_control_infer.py `
  --model-format dynamic `
  --model-path $Params `
  --dataset-dir $CruiseDataset `
  --eval-sources $ValSource `
  --split eval `
  --device gpu:0
~~~

### 静态模型检查

静态模型参数传模型前缀，不要附加 `.pdmodel`：

~~~powershell
$StaticPrefix = Join-Path $CruiseExport "cnn_lane"
& $Py scripts\nav_control_optional\check_nav_control_infer.py `
  --model-format static `
  --model-path $StaticPrefix `
  --dataset-dir $CruiseDataset `
  --eval-sources $ValSource `
  --split eval `
  --device gpu:0
~~~

### 连续帧和光照压力评估

连续帧分析会输出每帧预测、平滑预测、MAE、预测跳变、尖峰数量和每个 source 的摘要：

~~~powershell
& $Py scripts\nav_control_optional\check_nav_control_sequence.py `
  --model-format static `
  --model-path $StaticPrefix `
  --dataset-dir $CruiseDataset `
  --eval-sources $ValSource `
  --split eval `
  --smooth-window 5 `
  --output-dir .\tmp\nav_sequence_check
~~~

光照压力评估对图片生成固定的离线变体：

~~~powershell
& $Py scripts\nav_control_optional\evaluate_nav_control_robustness.py `
  --model-format static `
  --model-path $StaticPrefix `
  --data-dir $ValSource `
  --output-dir .\tmp\nav_light_stress\low_light `
  --illumination-variant low_light
~~~

支持的变体：

~~~text
original low_light overexposure low_contrast
left_shadow right_shadow center_glare warm_cast cool_cast
~~~

要完整跑一轮，逐个变更 `--illumination-variant`，不要把多种变体名写成一个参数。评估输出包括 `robustness_report.json`、`robustness_report.md` 和 `frame_predictions.csv`。这些变体是可重复的压力测试，不等价于真实赛道上的太阳直射、灯光反射、摄像头自动曝光或紫色区域画面。

### 巡航结果如何选

推荐按以下顺序保存证据：

1. 先确认验证集按完整 session 隔离，没有相邻帧泄漏。
2. 比较全局 MAE、弯道 MAE、过渡段 MAE、hard 样本 MAE。
3. 查看预测相邻帧差值的 p95、最大值、尖峰数和越界数。
4. 查看九种光照变体的误差变化，不只看原图平均 MAE。
5. 用同一套上位机控制参数，在同一赛道做旧模型/新模型 A/B。

静态模型通过检查不代表摄像头位置、视野外赛道区域、紫色停车区或停车动作已经解决。巡航模型的输出只包含偏差和兼容用的零值，停车状态机和速度策略属于上位机。

## 公开模型

仓库内有两份可直接取得的模型包：

~~~text
deploy/public_models/nav_control_lane_set7_20260808_balanced_v3_light_corner_epoch32.zip
deploy/public_models/target_det_27cls_20260818_epoch73_best_formal.zip
~~~

它们是 Git 工作树中的常规文件，不是 GitHub Release 附件。原始数据、训练缓存和更多训练输出按分卷归档在公开 Release [`data-archives-20260824`](https://github.com/indulgeuuu-del/smart_farm_models/releases/tag/data-archives-20260824) 中；其恢复规则、附件清单和 SHA256 入口见 `RELEASE_MANIFEST.json`。下载后必须先校验所有分卷，再解压。

SHA256：

~~~text
nav_control_lane_set7_20260808_balanced_v3_light_corner_epoch32.zip
69425C43321F416E25368FDA341A299229221E30E8BBAC203F92F8A0C9320C8C

target_det_27cls_20260818_epoch73_best_formal.zip
A5995DF6C8C46AE15BAC422C17B1F24A5F67C6A13A63A08880050FBE408FECD2
~~~

在 PowerShell 校验：

~~~powershell
Get-FileHash .\deploy\public_models\nav_control_lane_set7_20260808_balanced_v3_light_corner_epoch32.zip -Algorithm SHA256
Get-FileHash .\deploy\public_models\target_det_27cls_20260818_epoch73_best_formal.zip -Algorithm SHA256
~~~

单独读取随包提供的校验文件：

~~~powershell
Get-Content .\deploy\public_models\*.sha256
~~~

### 目标检测公开包

解压后，目标检测模型位于包内的 `target_det/`，应包含：

~~~text
model.pdmodel
model.pdiparams
model.pdiparams.info
infer_cfg.yml
label_list.txt
class_names.txt
SHA256.txt
~~~

将 `target_det/` 目录交给上位机前，先运行：

~~~powershell
Expand-Archive .\deploy\public_models\target_det_27cls_20260818_epoch73_best_formal.zip `
  -DestinationPath .\tmp\public_target_det `
  -Force

$PublicTarget = Get-ChildItem .\tmp\public_target_det -Recurse -Directory -Filter target_det |
  Select-Object -First 1
& $Py scripts\onboard\check_target_det_baseline_compat.py `
  --model-dir $PublicTarget.FullName `
  --expected-labels-file (Join-Path $PublicTarget.FullName "class_names.txt") `
  --json
~~~

公开 27 类包的类别顺序以包内 `class_names.txt` 和 `label_list.txt` 为准。复制到上位机时保持整个 `target_det/` 目录结构，不要只拿两个参数文件。

### 巡航公开包

解压后，巡航模型位于包内的 `nav_control_optional/`，包括动态参数、静态 Paddle 模型、导出元数据和 `.nb` 转换结果。包内还带有训练配置、评估报告、session 结果和光照压力报告。先看包内 README 和 `handoff_meta.json`，再核对上位机要求的文件名和输出顺序。

公开包是代表性候选，不是对任何硬件和赛道的永久保证。模型包里的评估数据、训练参数和 SHA256 是交付记录的一部分，不要重新压缩后丢失这些元数据。

## 复现一套最小可运行检查

只验证代码和现有元数据，不训练模型：

~~~powershell
$Repo = (Resolve-Path .).Path
$Py = Join-Path $Repo ".venv\Scripts\python.exe"

& $Py --version
& $Py -m pip check
& $Py -m compileall -q scripts tests
& $Py -m unittest discover -s tests -q
& $Py scripts\release\verify_public_repository.py --repo-root .
& $Py scripts\common\check_mandatory_data_layout.py --datasets-root .\datasets
~~~

如果本地已经准备好 00 的原始图和 Labelme JSON：

~~~powershell
& $Py scripts\lane_seg\check_lane_seg_dataset.py --dataset-dir .\datasets\00_lane_seg
& $Py scripts\lane_seg\split_dataset.py --dataset-dir .\datasets\00_lane_seg
& $Py scripts\lane_seg\export_seg_masks.py --split train
~~~

如果本地已经准备好一个增强后的 01 数据集：

~~~powershell
& $Py scripts\detection\check_det_coco_annotations.py `
  --task target_det `
  --image-dir .\datasets\01_target_det\<dataset>_aug\images `
  --coco-json .\datasets\01_target_det\<dataset>_aug\annotations\instance_train.json `
  --coco-json .\datasets\01_target_det\<dataset>_aug\annotations\instance_val.json `
  --coco-json .\datasets\01_target_det\<dataset>_aug\annotations\instance_test.json `
  --allow-custom-classes
~~~

如果本地已经准备好 08 的 train/val source：

~~~powershell
& $Py scripts\nav_control_optional\check_nav_control_data.py `
  --record-dir .\datasets\08_nav_control_optional\paddle_custom\<dataset>\train `
  --record-dir .\datasets\08_nav_control_optional\paddle_custom\<dataset>\val `
  --required-splits train,val
~~~

这些命令只在对应图片和元数据已经存在时才会通过。不要用空目录、伪造 JSON 或随机生成图片来替代正式数据检查。

## 版本、数据和许可记录

每一轮正式训练至少保存以下内容：

~~~text
数据集来源和文件清单
类别表及其顺序
train/val/test 划分方式和随机种子
离线增强名称、参数、变体数量和随机种子
配置文件副本
Python、Paddle、PaddleDetection 提交号和 GPU/驱动版本
训练命令、输出目录和最终退出码
best epoch 及选择指标
导出命令、导出文件清单和 SHA256
图片/连续帧/光照评估报告
Jetson 和上位机的运行模式、输入尺寸、阈值和动态 shape
实车赛道、光照、控制参数和 A/B 结果
~~~

原始图片、训练缓存和完整 checkpoint 不进入 Git 工作树。公开模型包只保留已经整理好的交付文件；归档范围、已知缺口和恢复边界见 `DATA_MODEL_PROVENANCE.md` 与 `RELEASE_MANIFEST.json`。数据是否可以公开，要同时遵守原始数据、图片、预训练权重、PaddleDetection 和 PaddlePaddle 的授权条件。仓库源码采用 MIT License，详见 `LICENSE`。

## 常见问题

### `pip` 安装了包，但脚本仍然找不到

不要只输入 `python`，先确认当前解释器：

~~~powershell
& $Py -c "import sys; print(sys.executable)"
& $Py -m pip show paddlepaddle-gpu
~~~

`pip` 和 `python -m pip` 可能属于不同 Python。本文所有安装命令都使用 `python -m pip` 的显式解释器形式。

### `paddle.is_compiled_with_cuda()` 为 False

检查是否误装了 CPU wheel、Python 是否为 64 位、显卡驱动是否正常、CUDA wheel 是否匹配。卸载错误分支后重新安装，不要在代码中把设备强行写成 `gpu:0` 试图绕过检查。

### `ppdet` 或 `infer.py` 找不到

目标检测脚本只接受锁定源码或显式路径。优先安装锁定版本后设置并验证：

~~~powershell
$PaddleDet = Join-Path $Repo "third_party\PaddleDetection"
.\env\install_paddledetection.ps1 -Destination $PaddleDet
$env:SMART_FARM_PADDLEDET_ROOT = (Resolve-Path -LiteralPath $PaddleDet).Path
Test-Path (Join-Path $PaddleDet "tools\train.py")
Test-Path (Join-Path $PaddleDet "deploy\python\infer.py")
& $Py -c "import sys; sys.path.insert(0, r'$PaddleDet'); import ppdet; print(ppdet.__file__)"
~~~

也可以在命令中通过 `--paddledet-root` 指定一个已核验目录。缺少 `deploy/python/infer.py` 时，数据检查仍可运行，但目标检测图片推理不能声称通过。

### 正式目标检测训练拒绝未增强目录

这是有意设置的保护。对 train 执行 `augment_target_det_coco.py`，确认输出中有 `augmentation_meta.json`，再传增强目录。只有单批 smoke test 或有明确记录的消融实验，才使用 `--allow-non-augmented-dataset`。

### 类别数或类别顺序不对

逐项比较 COCO `categories`、`class_names.txt`、`label_list.txt`、配置中的 `num_classes` 和上位机类别表。26 类和 27 类模型不能互换；公开 27 类模型新增 `green pepper`，但类别表顺序仍必须以模型包文件为准。

### 巡航验证误差异常好

先检查 train 和 val 是否来自同一 session 的相邻帧。如果有泄漏，重新使用 session 模式准备数据。再确认验证集没有使用训练增强、没有被 `drop_tail_frames` 错误删除，以及 `split_meta.json` 里的记录数量和 session 与实验计划一致。

### 巡航 `state` 的速度去了哪里

当前模型只回归 `state[2]` 偏差，输出第二列固定为零以兼容旧接口。速度字段不是第二个模型头。紫色区域停车、减速和停止条件属于上位机控制状态机；要让模型学习速度，必须先改变模型输出契约、数据标签和上位机消费逻辑，不能只在 README 或 YAML 中改名字。

### `eval_onboard_samples.py` 不能给出 AP

这个工具面向未标注的板端图片。它只能证明图片被发现、模型文件齐全，并在明确请求时执行推理；没有 ground truth 时不能计算 AP、mAP 或准确率。要评估指标，必须使用带完整 COCO 标注且类别表一致的验证/测试集。

### TensorRT FP16 在电脑能跑，Jetson 不能跑

逐项核对导出版本、TensorRT 版本、CUDA/驱动、模型输入 shape、动态 shape 范围、模型文件完整性和上位机预处理。删除 TensorRT cache 后重新构建只能作为排查步骤，不能把“成功构建一次”当成稳定交付。最终必须在目标 Jetson 上用上位机实际入口验证。

## 最终验收清单

### 代码和环境

- [ ] `git status` 只有预期的本地输出，没有把数据或 checkpoint 误加入；
- [ ] 三个虚拟环境的解释器路径明确，没有混用；
- [ ] 主环境 `pip check` 通过；
- [ ] `compileall` 通过；
- [ ] `unittest discover` 通过；
- [ ] 当前脚本入口的 `--help` 全部通过。

### 数据

- [ ] 00 的 Labelme 类别和 mask 目录通过静态检查；
- [ ] 01 的 COCO 三个 split、图片、bbox、category id 全部通过检查；
- [ ] 01 增强只作用于 train，val/test 未被污染；
- [ ] 08 的图片路径、标签范围、session 划分和 `split_meta.json` 已核对；
- [ ] 机器标注数据已人工抽检，疑似错标、漏标、重复框和空图片有处理记录。

### 模型

- [ ] 训练命令、配置、数据目录、外部源码提交号和随机种子已保存；
- [ ] best epoch 的选择依据明确，不能只写“效果好”；
- [ ] 导出文件成对存在，类别表和输入尺寸一致；
- [ ] SHA256 已计算并随交付物保存；
- [ ] 电脑 Paddle 图片推理已完成；
- [ ] TensorRT FP16 或 PaddleLite 的目标设备验证已完成；
- [ ] 上位机实际入口和固定控制参数下完成 A/B；
- [ ] 实车赛道、光照、摄像头位置和结果已记录。

## 责任边界

本仓库可以对数据格式、模型训练、导出物结构和离线评估负责；以下内容必须在配套工程或硬件上确认：

- 上位机摄像头是否打开、画面是否显示检测框和 FPS；
- 目标检测结果的后处理、重复识别策略和任务状态机；
- 巡航输出怎样变成底盘 PWM、舵机角度或 IMU 保持；
- 紫色区域的减速和停车；
- Jetson 的 TensorRT FP16、PaddleLite `.nb` 和真实摄像头时序；
- 底盘横移、机械臂、舵机、串口和烧录；
- 最终比赛成绩。

模型工程的最后一个文件是模型；整车系统的最后一个结论必须来自整车。
