# AI 架构技术路线与可行性评估报告

## 1. 核心功能可行性与难点分析 (Feasibility Analysis)

基于 `data_insights_readme.md` 的产品需求，以下功能点存在较高的技术挑战或实现风险：

### 1.1 高风险/高难度功能
*   **F1.2 隐性情绪计算 (Implicit Sentiment & Energy)**
    *   **难点：** 传统的 Sentiment Analysis 模型（如基于 IMDb/SST-2 训练）擅长识别显性的褒贬词（如 "happy", "bad"），但难以理解“短句+负面词”所代表的**低能量状态**。例如，“今天只写了50字。”在传统模型看来可能是中性，但在本场景下代表“低效/挫败”。
    *   **挑战：** 缺乏特定场景的标注数据（Domain Adaptation）。
    *   **对策：** 不单纯依赖 BERT 输出，需引入**规则引擎 (Heuristic Rules)** 辅助（如结合句长、标点密度、特定否定词表）进行加权打分。

*   **F2.4 隐性习惯关联 (Implicit Habit Association / Apriori)**
    *   **难点：** 用户的日记数据是稀疏的（Sparse Data）。Apriori 算法在数据量不足时难以挖掘出有统计显著性的规则，容易产生“伪关联”。
    *   **对策：** 初期降级为简单的**相关性分析 (Correlation Analysis)**，或基于预设模板的规则匹配，待数据积累后再引入复杂挖掘算法。

### 1.2 中等难度功能
*   **F2.1 生活重心热力聚类 (K-Means)**
    *   **难点：** 在端侧对不断增长的 Embedding 向量进行实时 K-Means 聚类，计算开销会随数据量线性/超线性增长。
    *   **对策：** 采用增量式聚类 (Incremental Clustering) 或定期（如每周）在充电闲置时重算，而非实时计算。

### 1.3 性能约束挑战 (Non-functional)
*   **40MB 模型体积 & 200ms 延迟：**
    *   标准 BERT-Base 约 400MB，必须使用 **MobileBERT** 或 **DistilBERT** 并配合 **Int8 量化** 才能压到 40MB 以下。
    *   iOS 上的 CoreML Neural Engine 优化是必须的，否则 CPU 推理很难达到 200ms/条。

---

## 2. 技术选型 (Technology Stack)

### 2.1 核心模型 (The Brain)
*   **架构：** MobileBERT (Google) 或 TinyBERT。
*   **格式：** CoreML (iOS) / TFLite (Android/Backup)。
*   **训练策略：**
    1.  **Base Model:** 使用通用中文语料预训练的 MobileBERT。
    2.  **Fine-tuning:** 构建一个小型的“目标管理/日记”数据集（需合成或人工标注约 1k-5k 条数据）进行微调，重点优化多标签分类和实体抽取。

### 2.2 数据存储与向量检索 (Memory)
*   **结构化数据：** SQLite (使用 Room 或 CoreData)。
*   **向量数据 (Embeddings)：**
    *   方案 A (推荐): **SQLite-vss** (若支持端侧编译) 或简单的 FlatBuffer 文件存储（数据量 < 10k 条时，暴力搜索亦可）。
    *   方案 B: iOS 原生 **NaturalLanguage 框架** 的 Embedding 接口（如果不需自定义模型）。*建议仍使用自定义模型以保证多端一致性和微调能力。*

### 2.3 任务调度 (Scheduler)
*   **iOS:** `BGTaskScheduler` (后台任务) + `Combine` (异步流)。
*   **策略：** 写入时仅做轻量级处理（正则），重量级 NLP (BERT) 放入后台队列或下次启动时处理。

---

## 3. 架构设计 (Architecture Design)

```mermaid
graph TD
    User[用户输入] --> InputQueue[输入缓冲队列]
    InputQueue --> |异步/闲置| AI_Engine[AI 洞察引擎]
    
    subgraph "AI 洞察引擎 (Local Device)"
        Pre[预处理 & 正则] --> Encoder[MobileBERT Encoder]
        Encoder --> |Vectors| Feature_Ext[特征提取层]
        
        Feature_Ext --> Class[分类头 (F1.1)]
        Feature_Ext --> NER[实体抽取头 (F1.3)]
        Feature_Ext --> Sentiment[情绪回归头 (F1.2)]
        
        Class --> DB[(本地 SQLite)]
        NER --> DB
        Sentiment --> DB
        Encoder --> |Embedding| VecDB[(向量存储)]
    end
    
    subgraph "分析服务 (Analysis Service)"
        Trigger[定时/充电触发] --> Analyzer[统计分析器]
        VecDB --> Cluster[K-Means 聚类 (F2.1)]
        DB --> Trend[趋势/关联分析 (F2.2-2.4)]
        
        Cluster --> Dashboard[前端看板]
        Trend --> Dashboard
    end
```

---

## 4. 开发实施规划 (Implementation Roadmap)

### Phase 1: 基础设施与 MVP (对应 PRD Phase 2.1)
*   **目标：** 模型跑通，能分类，能存数据。
*   **任务：**
    1.  **模型转换：** 将 HuggingFace 的 `google/mobilebert-uncased` (或中文版) 转换为 CoreML 格式，进行 Int8 量化。
    2.  **推理管线：** 封装 `NLPEngine` 单例，实现 `text -> model -> labels` 的流程。
    3.  **数据库变更：** 扩展 `Goal` 表，增加 `embedding` (blob), `sentiment_score` (float), `tags` (string) 字段。
    4.  **UI：** 简单的饼图展示分类占比。

### Phase 2: 向量能力与聚类 (对应 PRD Phase 2.2)
*   **目标：** 让数据“聚”起来，发现主题。
*   **任务：**
    1.  **向量存储：** 实现基于文件的向量读写器。
    2.  **聚类算法：** 实现轻量级 K-Means (K=5~10)。
    3.  **UI：** 词云/气泡图开发。

### Phase 3: 高级洞察与优化 (对应 PRD Phase 2.3)
*   **目标：** 情绪分析与建议。
*   **任务：**
    1.  **情绪算法优化：** 引入规则引擎修正 BERT 的情绪打分。
    2.  **关联挖掘：** 实现简单的共现矩阵分析 (Co-occurrence Matrix) 替代复杂的 Apriori。
    3.  **性能调优：** 引入 `BGTaskScheduler`，确保主线程零卡顿。

---

## 5. 数据隐私与模型迭代策略
由于承诺 **Privacy First**，我们无法回传数据训练。
*   **策略：** **Federated Analytics (联邦统计)** 的简化版。
*   **实现：** 仅回传“模型置信度分布”、“标签修正率”等统计指标（不含文本），用于监控模型在全体用户上的泛化能力。如果普遍效果差，则在云端使用通用公开数据集重新训练模型并下发更新（OTA Update）。
