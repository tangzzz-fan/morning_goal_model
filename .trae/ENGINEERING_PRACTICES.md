# Morning Goal Model - 卓越工程实践指南

本文档定义了本项目的核心开发流程与规范，旨在解决“上下文丢失”、“目标模糊”与“进度不可控”的问题。所有开发活动均需遵循本指南。

## 1. 核心工作流 (The Core Workflow)

为了保持高效并避免上下文丢失，我们采用 **"Context-First Development" (上下文优先开发)** 模式。

### 🔄 标准交互循环
1.  **上下文恢复 (Context Recovery)**: 
    *   在开始任何新任务前，**必须** 阅读 `.trae/PROJECT_CONTEXT.md`。
    *   确认当前所处的里程碑 (Milestone) 和活跃任务 (Active Task)。
2.  **计划先行 (Plan First)**:
    *   使用 `TodoWrite` 工具列出详细的执行步骤。
    *   对于复杂变更，先在对话中简述方案，再执行代码修改。
3.  **原子化执行 (Atomic Execution)**:
    *   每个 Todo 项对应一个具体的代码变更或验证动作。
    *   保持变更的“原子性”，避免一次性修改过多无关文件。
4.  **即时验证 (Verify Immediately)**:
    *   **无验证不交付**。任何代码修改后，必须运行对应的测试脚本或验证命令。
    *   验证结果（Logs, Screenshots, Metrics）必须在回复中体现。
5.  **状态锚定 (Anchor State)**:
    *   **关键规则**：在结束当前对话回合（Turn）或完成一个大任务后，**必须** 更新 `.trae/PROJECT_CONTEXT.md`。
    *   记录：已完成什么、下一步做什么、已知阻碍。

## 2. 文档体系 (Documentation System)

我们使用轻量级、高维护性的文档结构：

*   **`.trae/PROJECT_CONTEXT.md` (项目上下文锚点)**
    *   **唯一真理源**：记录项目的实时状态。
    *   **包含**：当前阶段、最新进展、技术决策栈、环境变量备忘。
    *   **更新频率**：每次重大变更后。

*   **`development_plan.md` (长期路线图)**
    *   **作用**：宏观的项目里程碑规划。
    *   **更新频率**：里程碑完成时。

*   **`docs/` (技术知识库)**
    *   **作用**：沉淀通用的技术方案、API文档、架构设计。
    *   **原则**：只存有长期价值的内容，避免存临时笔记。

## 3. 开发规范 (Development Standards)

### 3.1 代码变更原则
*   **优先修改，慎重创建**：优先基于现有文件修改，避免文件爆炸。
*   **保持清洁**：删除不再使用的代码和文件（Dead Code Elimination）。
*   **自解释**：代码注释解释“为什么”而不是“是什么”。

### 3.2 验证标准 (Verification Standards)
所有功能交付必须满足 **DoD (Definition of Done)**：
1.  代码已实现。
2.  **静态检查通过**：无明显的 Syntax Error 或 Linter Error。
3.  **功能验证通过**：
    *   **Python/Model**: 运行过对应的 `scripts/` 或 `tests/`，并输出了成功的 Log。
    *   **iOS**: 编译通过，或通过脚本模拟了输入输出。
4.  **文档已更新**：`PROJECT_CONTEXT.md` 已反映最新状态。

## 4. 常用命令备忘 (Cheatsheet)

*   **激活环境**: `source activate_venv.sh`
*   **运行所有测试**: `bash scripts/run_tests.sh`
*   **运行质量测试**: `bash scripts/run_tests.sh -t quality`
*   **导出 CoreML**: `python src/export/export_coreml.py`

---
*Created by Trae AI Assistant to ensure high-velocity, high-quality delivery.*
