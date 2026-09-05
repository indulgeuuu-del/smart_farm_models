# 更新记录

仓库公开整理时重建了 Git 历史。以下记录从公开初始版本开始。

## [Unreleased]

- 完善公开仓库检查项：校验结构化 Issue Form、Pull Request 模板、CI/CodeQL 必要步骤和固定提交的 GitHub Action；补充本地审查与训练监控产物的忽略规则。
- 增加 CODEOWNERS 和结构化 Issue 入口，并让公开仓库校验器检查 GitHub 治理文件。
- 增加可续传、逐附件 SHA256 校验的公开 Release 下载器，并将其纳入仓库校验和单元测试。
- 固定 CI 与 CodeQL 使用的 GitHub Actions 提交，并增加依赖 profile 的静态兼容检查。
- 修正 PaddleDetection 依赖安装流程，过滤废弃的 `sklearn==0.0` shim，并在解析时约束到仓库锁定 profile。
- 增加跨平台公开仓库 smoke 检查入口，补充 Linux 复现说明，并写明已发布模型离线指标的适用范围。
- 固定 PaddleDetection 上游提交并增加安装脚本。
- 移除对私有本机目录的目标检测推理依赖。
- 增加公开模型结构与 SHA256 验证器、CI、依赖更新和治理文件。
- 将无标注板端样本工具改为真实预检入口，明确它不产出 AP。

## [2026-08-25] Initial public release

- 公开 00 车道分割数据工具、01 目标检测工作流和 08 巡航回归工作流。
- 在 `deploy/public_models/` 提供一份 27 类目标检测交付包和一份巡航交付包，并附 SHA256。
