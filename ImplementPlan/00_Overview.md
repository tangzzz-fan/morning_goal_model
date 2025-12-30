# Morning Goal Insight System v2.0 Implementation Plan Overview

This folder contains the detailed step-by-step implementation plan for the **Morning Goal Insight System v2.0**, based on the PRD and Technical Roadmap.

## Plan Structure

The plan is divided into phases corresponding to the roadmap:

*   **[01_Phase2.1_MVP_Model_and_Infra.md](./01_Phase2.1_MVP_Model_and_Infra.md)**:
    *   Focus: Model training, conversion (CoreML), and basic Python-side validation.
    *   Goal: Produce a deployable `mlpackage` with Multi-label classification and Sentiment analysis.
    
*   **[02_Phase2.1_MVP_iOS_Integration.md](./02_Phase2.1_MVP_iOS_Integration.md)**:
    *   Focus: iOS app infrastructure, Core Data updates, Inference Engine integration, and Basic UI.
    *   Goal: App can run the model, store results in SQLite, and show basic statistics.

*   **[03_Phase2.2_Intelligence_Engine.md](./03_Phase2.2_Intelligence_Engine.md)**:
    *   Focus: Vector storage (Embeddings), K-Means Clustering, and "Insight" visualization.
    *   Goal: Enable semantic clustering and "Focus Bubbles".

*   **[04_Phase2.3_Pro_Features.md](./04_Phase2.3_Pro_Features.md)**:
    *   Focus: Advanced algorithms (Burnout Alert, Habit Association), Rule Engines, and Optimization.
    *   Goal: Complete the "Personal Coach" experience.

## Verification & Tracking

Each step in the plan includes a `## Verification` section.
*   **Python/Model tasks**: Verify using specific scripts (e.g., `python scripts/test_model.py`).
*   **iOS tasks**: Verify using XCTest or specific UI checks.
*   **Data tasks**: Verify using SQL queries or inspection tools.
