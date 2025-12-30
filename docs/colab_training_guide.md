# Google Colab 训练指南

本指南将详细介绍如何在 Google Colab 上使用 `MorningGoalModel` 项目进行多任务模型的训练。

## 1. 准备工作：上传项目到 Google Drive

在开始之前，你需要将本地的项目文件上传到 Google Drive。

### 步骤
1.  **登录 Google Drive**: 使用你的 Google 账号登录 [drive.google.com](https://drive.google.com)。
2.  **创建文件夹**: 在 "我的云端硬盘 (My Drive)" 根目录下创建一个新文件夹，命名为 `MorningGoalModel`。
    *   *注意：如果你更改了文件夹名称，请记得修改 Notebook 中的路径配置。*
3.  **上传文件**: 将你本地项目中的以下关键文件和文件夹上传到刚才创建的 `MorningGoalModel` 文件夹中：
    *   `src/`: **必须**。包含所有源代码。
    *   `data/`: **必须**。包含训练数据。
        *   确保 `data/processed/` 目录下有 `train_multitask.csv`, `val_multitask.csv`, `test_multitask.csv` 这三个文件。
    *   `configs/`: **推荐**。包含配置文件。
    *   `models/`: **可选**。如果你想基于之前训练好的模型（如 `distill_student`）继续训练，请上传 `models/trained/distill_student` 文件夹。如果你打算使用 `bert-base-chinese` 从头微调，则不需要上传此文件夹。
    *   `colab/Multitask_Training.ipynb`: **必须**。这是我们刚才生成的专用训练笔记本。

**上传后的 Drive 目录结构应如下所示：**
```
My Drive/
└── MorningGoalModel/
    ├── src/
    ├── data/
    │   └── processed/
    │       ├── train_multitask.csv
    │       ├── val_multitask.csv
    │       └── test_multitask.csv
    ├── models/ (可选)
    └── colab/
        └── Multitask_Training.ipynb
```

## 2. 打开并配置 Colab Notebook

1.  **打开 Notebook**:
    *   在 Google Drive 中找到 `MorningGoalModel/colab/Multitask_Training.ipynb`。
    *   右键点击 -> 打开方式 -> Google Colaboratory。
    *   或者直接访问 [colab.research.google.com](https://colab.research.google.com)，选择 "File" -> "Open notebook" -> "Google Drive"，然后选择该文件。

2.  **启用 GPU 加速**:
    *   在 Colab 顶部菜单栏，点击 **修改 (Edit)** -> **笔记本设置 (Notebook settings)**。
    *   在 **硬件加速器 (Hardware accelerator)** 下拉菜单中选择 **GPU** (通常是 T4 GPU)。
    *   点击 **保存 (Save)**。

## 3. 执行训练步骤

按照 Notebook 中的单元格顺序依次执行：

### 第一步：安装依赖
运行第一个代码块，它会自动安装 `transformers`, `datasets` 等必要的 Python 库。
```python
!pip install transformers datasets accelerate scikit-learn pandas
```

### 第二步：挂载 Google Drive
运行第二个代码块。
```python
from google.colab import drive
drive.mount('/content/drive')
```
*   系统会弹出一个窗口请求访问权限，点击 **连接到 Google Drive (Connect to Google Drive)** 并确认。

### 第三步：设置项目路径
运行第三个代码块。它会将 Python 的运行目录切换到你上传的文件夹。
*   **关键点**：如果你在上传时使用了不同的文件夹名称（不是 `MorningGoalModel`），请修改 `PROJECT_PATH` 变量。
```python
PROJECT_PATH = '/content/drive/MyDrive/MorningGoalModel'
```

### 第四步：导入训练模块
运行此块以验证环境配置是否正确。如果看到 "Successfully imported training module"，说明一切正常。

### 第五步：配置训练参数
在这里你可以调整训练的超参数。
*   `base_model`:
    *   如果你上传了 `models/trained/distill_student`，可以将其设置为 `"models/trained/distill_student"`。
    *   否则，保持默认的 `"bert-base-chinese"`，它会自动从 Hugging Face 下载预训练模型。
*   `batch_size`: 默认 32。如果遇到显存不足 (OOM) 错误，可以调小到 16 或 8。
*   `epochs`: 默认 5 轮。
*   `lr`: 学习率，默认 2e-5。

### 第六步：开始训练
运行此块开始训练。
*   你可以看到实时的训练进度条、Loss 变化和验证集指标。
*   训练完成后，模型会自动保存到 Google Drive 的 `models/trained/multitask_model_colab` 目录下。

## 4. 获取训练结果

训练结束后，你可以在 Google Drive 的文件夹中找到以下产物：
1.  **模型权重**: `models/trained/multitask_model_colab/` 下的 `pytorch_model.bin` (或 `model.safetensors`)。
2.  **配置文件**: `config.json`, `tokenizer.json` 等。
3.  **评估报告**: `metrics_test.json` 包含了模型在测试集上的 F1 分数和准确率。

## 5. 模型推理与测试

训练完成后，你可以直接在 Notebook 中测试模型的效果。

### 执行推理
运行 Notebook 中的 **"7.1 Run Inference"** 代码块。
*   脚本会加载刚刚训练好的模型。
*   对预设的测试文本（如 "今天完成了5公里跑步"）进行分类和情感分析。
*   输出结果将显示预测的主题（如 "运动"）和情感（如 "积极"）及其置信度。

### GPU 资源说明
*   **训练阶段**: **强烈建议使用 GPU**。BERT 模型的训练计算量大，使用 T4 GPU 可以将训练时间从几小时缩短到几分钟。
*   **推理阶段**:
    *   **批量推理**: 如果需要处理大量数据（如几千条文本），建议保持 GPU 开启，速度会快很多。
    *   **单条/少量测试**: CPU 也可以胜任，速度在可接受范围内（每条约 100-200ms）。
    *   *注意：Colab 的 GPU 是有使用限额的，如果只做简单的测试，可以考虑在训练结束后断开运行时，重新连接并选择 "None" (CPU) 以节省 GPU 配额。*

## 常见问题 (FAQ)

**Q: 遇到 `ModuleNotFoundError: No module named 'src'` 怎么办？**
A: 请检查第三步中的 `PROJECT_PATH` 是否正确。确保你的 Drive 中确实存在该路径，且该路径下直接包含 `src` 文件夹。

**Q: 训练速度很慢？**
A: 请确认你是否开启了 GPU 模式（参见第 2 节）。使用 CPU 训练 BERT 会非常慢。

**Q: Google Drive 空间不足？**
A: 训练产生的 Checkpoint 文件可能较大。你可以在 `TrainingArgs` 中修改 `save_total_limit` 参数来限制保存的 Checkpoint 数量，或者定期清理 Drive 中的旧模型文件。
