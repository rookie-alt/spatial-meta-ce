# server68 CE 部署验证

日期：2026-11-23。状态：`prepared_no_training`。

本记录同步实际部署结果。server68 使用账号 `liushuaixi`，所有项目文件位于 `/home/liushuaixi/weilin`。本轮创建的 `/gly/weilin` 已删除，`/gly` 当前为空父目录。部署、环境、路径认证和未签 dry-run 已完成；没有启动训练或模型推理。

## 结果

| 项目 | 结果 |
|---|---|
| GitHub Release | `module26-migration-20261123`，30/30 附件大小和 SHA256 通过 |
| 原始归档 | 634/634 文件、17,012,110,734 bytes 完整回读通过 |
| 归档 SHA256 | `9077240a3e1b560aed7eb0366049f6195629f26c647ae04224028c2613aaea73` |
| 训练代码 | 提交 `26d7056935e445c5c3d27926fd5e90e6dd5f4e06`；37/37 冻结文件 SHA 未变 |
| 输入认证 | 30 个库、1,307,239 行；失败库、unsupported 和重复冲突均为 0 |
| mapping | 30 份重新生成的 `mapping.jsonl` 与原映射内容一致 |
| 固定计划 | 三份计划的字节和 SHA 未变，各 96,000 episodes |
| Python 环境 | Python 3.11.18、PyTorch 2.1.0+cu121、NumPy 1.26.1、SQLGlot 18.17.0 |
| 未签准备 | 15 份训练配置，全部 `authorized=false` |
| dry-run | `dry_run=true`、`executed=false`、`authorization=not_provided` |
| 训练与推理 | `training_run=false`、`model_predictions_run=false`；训练输出目录不存在 |
| 磁盘 | 记录时可用 5,366,756,413,440 bytes |

## 路径

- 项目根目录：`/home/liushuaixi/weilin`
- 训练代码：`/home/liushuaixi/weilin/module26-prep-20261118/code-training`
- Python 环境：`/home/liushuaixi/weilin/module26-runtime/venv`
- 激活入口：`source /home/liushuaixi/weilin/activate-ce.sh`
- 迁移包与证据：`/home/liushuaixi/weilin/module26-migration-20261123`
- 15 份未签配置：`/home/liushuaixi/weilin/module26-prep-20261118/training-ready-server68`

活跃配置和生成配置不再包含 `/gly/weilin`。历史归档和原始证据保留该路径字符串，用于记录来源。

## 限制

迁移后的 23+3 source config 从冻结的 26+0 配置和固定划分规则重建，不是历史 protocol-122 source config 的字节验证副本。CUDA 小算例未运行，因为当时 8 张 GPU 均忙碌；`cuda_small_compute_passed=null`，不能据此宣称 GPU 已就绪。

## 证据

- [server68-deployment-verification.json](server68-deployment-verification.json)
- [server68-download-verification.json](server68-download-verification.json)
- [server68-paths-verification.json](server68-paths-verification.json)
- [server68-environment-verification.json](server68-environment-verification.json)
- [server68-dry-run-result.json](server68-dry-run-result.json)
- [server68-relocated-data-verification.json](server68-relocated-data-verification.json)
- [server68-deployment-status.json](server68-deployment-status.json)
- [server68-migration-helper-tests.log](server68-migration-helper-tests.log)
- [server68-preparation.log](server68-preparation.log)
