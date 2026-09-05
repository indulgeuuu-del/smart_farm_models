# 数据与模型来源边界

## Git 工作树中有什么

公开 Git 工作树包含代码、训练配置、COCO/JSONL 等元数据、类别表、数据划分记录、模型交付 ZIP 和对应 SHA256。`deploy/public_models/` 当前包含 27 类目标检测和巡航模型的可用交付包。`scripts/release/verify_public_repository.py` 会检查它们的哈希和必要文件结构。

## Git 工作树中没有什么

原始图片、训练缓存、完整 checkpoint、运行日志及第三方 PaddleDetection 源码没有放进 Git。没有原始图片时，COCO 标注 JSON 只能说明历史数据契约，不能单独复现训练样本或历史指标。没有完整 checkpoint 时，也不能从公开 Git 工作树继续历史训练。

## 数据归档 Release

`RELEASE_MANIFEST.json` 记录了 2026-08-24 创建、2026-08-28 正式发布的 [`Public data archives 2026-08-24`](https://github.com/indulgeuuu-del/smart_farm_models/releases/tag/data-archives-20260824) 数据归档。它由多个 `part-*.tar.gz` 分卷、`SHA256SUMS.txt`、`source_manifest.tsv`、`part_plan.csv` 和 `scope.json` 构成。该 Release 已公开，但公开可下载不等于归档内容已在每一台机器上重新解压和验证。

这个历史 Release 产生于公开整理规则完善以前。`scope.json` 保留了生成机器的本机路径，Release 说明也把运行日志列入了归档范围。它是不可变的历史交付物，不要继续沿用这套打包流程。重新生成公开归档时，使用 `scripts/release/build_public_data_release.ps1`；当前脚本不写本机路径，并明确排除 `logs/` 下文件名包含 `progress` 的 Markdown 进度日志。若要发布清理后的历史归档，应创建新的 tag 和新的完整校验清单。

使用归档前必须完成以下检查：下载全部附件；用 `SHA256SUMS.txt` 校验所有分卷；确认解压后的路径和大小与 `source_manifest.tsv` 一致；确认本地使用范围符合原始数据、图片和设备记录的授权边界。后续重新打包或替换附件时，必须建立新的 tag 和新的 SHA256 清单，不能覆写既有交付物。

## 使用和再发布

仓库源码采用 MIT License；这不自动授予任何原始图片、训练数据、预训练权重、第三方模型或第三方源码的再发布权。使用者应自行确认数据采集授权、人物与场地信息、竞赛条款、PaddlePaddle/PaddleDetection 的许可和目标设备软件许可。重新打包模型时必须保留类别顺序、输入尺寸、配置、交付元数据和 SHA256。
