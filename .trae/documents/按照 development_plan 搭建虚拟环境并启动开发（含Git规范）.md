# 目标
根据 `development_plan.md` 的里程碑与“下一步计划”，在本地（Windows）创建/激活 Python 虚拟环境并开展 M1 阶段开发工作，同时建立可复用的 Git 规范与提交序列，确保每个步骤均可形成规范化提交。

## 文档与代码依据
- 计划文档: `development_plan.md`（下一步计划见第 91–96 行，参考：`development_plan.md:91`）
- 虚拟环境与依赖脚本：
  - `setup_venv.py`（创建 venv、安装依赖、生成激活脚本，参考函数：`setup_venv.py:39`、`setup_venv.py:61`、`setup_venv.py:75`、`setup_venv.py:130`、主流程 `setup_venv.py:230`）
  - 环境验证脚本：`environment_setup.py`（主流程参考：`environment_setup.py:89`）
- 依赖清单：`requirements.txt`
- 现有目录结构：`src/`（`data/`, `models/`, `training/`, `evaluation/`, `coreml/` 皆已初始化）

## 阶段A：虚拟环境与依赖
- 首选一键脚本（推荐）：
  - 在项目根目录执行：`python setup_venv.py`
  - 激活：`venv\Scripts\activate`（或脚本生成的 `activate_venv.bat`，见 `setup_venv.py:290-296` 的用法提示）
  - 验证环境：`python environment_setup.py`（生成 `environment_report.json`）
- 手动方案（若脚本失败）：
  - 创建：`py -3 -m venv venv`
  - 激活：`venv\Scripts\activate`
  - 安装依赖：`python -m pip install --upgrade pip`；`pip install -r requirements.txt`
  - 验证环境：`python environment_setup.py`

## 阶段B：Git 工作流与提交规范
- 分支模型（Git Flow 精简版）：
  - `main`：稳定发布线；`develop`：集成线
  - 里程碑/特性分支：`feature/m1-data-prep`、`feature/m2-pretrain` 等
- 提交格式（Conventional Commits）：
  - `type(scope): subject`，type 常用：`feat`/`fix`/`docs`/`chore`/`refactor`/`test`/`build`
  - 示例：`feat(data): add raw dataset import pipeline`
- 标签规范：里程碑完成后打标签，如：`m1-ready`, `m2-baseline`；版本按 `v0.1.0` 语义化
- 基本命令：
  - 初始化/设置（如未初始化）：`git init`
  - 创建集成分支：`git checkout -b develop`
  - 创建特性分支：`git checkout -b feature/m1-data-prep`
  - 提交：`git add -A && git commit -m "type(scope): subject"`
  - 合并：`git checkout develop && git merge --no-ff feature/m1-data-prep`
  - 标签：`git tag -a m1-ready -m "M1 complete"`

## 阶段C：按里程碑M1推进的可提交步骤
- 步骤1：环境就绪（A 阶段完成）
  - 提交示例：`chore(env): create venv and install dependencies`（附 `environment_report.json`）
- 步骤2：数据收集（`data/raw/`）
  - 动作：收集 ≥10,000 条公开文本，保留来源与许可信息
  - 提交示例：`feat(data): add initial raw dataset sources and loaders`
- 步骤3：标注规范（文档与示例）
  - 动作：制定任务标签体系与边界案例；示例用法与一致性规则
  - 提交示例：`docs(annotation): add labeling guideline and examples`
- 步骤4：数据划分（70/20/10）
  - 动作：在 `src/data/` 编写划分脚本（可读性优先），输出到 `data/processed/`
  - 提交示例：`feat(data): implement train/val/test split with stratification`
- 步骤5：数据版本控制（DVC，可选但建议）
  - 动作：`dvc init`；配置远端存储；跟踪 `data/` 大文件
  - 提交示例：`build(dvc): initialize DVC and track data artifacts`
- 完成M1后：
  - 合并到 `develop`，打里程碑标签：`m1-ready`

## 阶段D：质量保障（每次提交前）
- 代码质量：
  - 格式化：`black .`
  - 静态检查：`flake8`、`mypy`（与 `requirements.txt` 一致）
  - 快速测试：`pytest -q`（为数据/函数添加基础用例）
- 提交原子性：一次仅做一件事；信息完整、可复现；必要时在提交正文说明数据来源与脚本入口

## 阶段E：后续M2~M4启动提示（概览）
- M2 预训练与微调：拉取 `bert-base-chinese`，仅微调末 3 层，产出 baseline（参考 `development_plan.md:23-29`）
  - 提交示例：`feat(model): add bert-base-chinese fine-tuning pipeline`
- M3 优化：蒸馏→量化（FP16/INT8）→剪枝；精度阈值控制（参考 `development_plan.md:33-38`）
- M4 CoreML 转换：`coremltools` + ONNX opset=13，精度差异<3%（参考 `development_plan.md:42-47`）

## Windows 兼容说明
- 激活优先使用 `venv\Scripts\activate`；若存在 `activate_venv.bat` 也可直接双击/执行
- 如需 Bash 脚本（`activate_venv.sh`），请在 Git Bash/WSL 中使用；当前平台优先 PowerShell/命令行

## 交付与验证
- 每个步骤完成后：
  - 本地验证（质量保障项）→ 形成规范化 Git 提交 → 推送到远端（如已配置）
  - 里程碑完成时合并到 `develop` 并打标签，以便后续 M2~M4 持续集成与回溯

请确认以上计划；确认后我将按该计划在本地执行环境搭建与开发推进，并在每个关键步骤产出规范的 Git 提交与说明。