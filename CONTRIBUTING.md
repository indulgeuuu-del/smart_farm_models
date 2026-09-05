# 参与贡献

先读根目录 `README.md`，确认改动属于 00、01 或 08 中的一条主线。仓库保存模型工程，不保存原始图片、训练缓存和完整 checkpoint。不要把个人采集目录、摄像头视频、设备日志、令牌、硬件地址或未经确认可公开的数据加入提交。

## 提交前的准备

1. 从 `main` 拉出独立分支，说明要解决的问题和验证范围。
2. 只改解决该问题必需的文件。配置、类别表、数据划分或模型输出契约的变化必须同时更新对应测试和 README。
3. 对目标检测训练，依次完成 COCO/类别表检查、仅训练集离线增强、增强后检查，再启动正式训练。不要用未增强数据冒充正式训练。
4. 对巡航训练，按完整 session 划分 train/val/test，不能把连续帧随机打散后宣称泛化。
5. 新模型交付必须带模型目录、类别表、输入尺寸、配置、SHA256 和离线验证记录；板端或实车结论要单独标明，不得用电脑结果替代。

## 本地检查

在已创建的 Python 3.10 环境中运行：

```powershell
$Py = Join-Path (Resolve-Path .) ".venv\Scripts\python.exe"
& $Py -m pip check
& $Py -m compileall -q scripts tests
& $Py scripts\common\check_supported_cli.py --repo-root . --python-executable $Py
& $Py scripts\common\smoke_public_repository.py --repo-root .
& $Py scripts\release\verify_public_repository.py --repo-root .
& $Py -m unittest discover -s tests -v
```

如涉及锁定的 PaddleDetection，再运行 `env/install_paddledetection.ps1`。只有明确设置 `SMART_FARM_RUN_PADDLEDET_INTEGRATION=1` 后，才运行依赖该 checkout 的集成测试。

## Pull Request

PR 说明至少写明：问题、修改范围、数据或模型契约是否改变、运行过的命令及结果、未验证项和硬件风险。不要提交二进制 checkpoint；公开模型应封装为独立 ZIP，附带 SHA256 与交付元数据。
