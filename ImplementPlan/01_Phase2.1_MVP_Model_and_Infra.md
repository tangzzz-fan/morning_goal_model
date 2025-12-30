# Phase 2.1: MVP Model & Infrastructure (Python/CoreML Side)

**Goal:** Establish the foundational AI capabilities: Multi-label classification, Basic Sentiment Analysis, and Entity Extraction, and export a quantifiable CoreML model.

## Step 1: Dataset Preparation for Multitask Learning
Refine the dataset to support Multi-label Classification (Topics) and Sentiment Regression/Classification.

*   **Action 1.1**: Update `src/data/add_sentiment_labels.py` (or create new `prepare_multitask_data.py`).
    *   Task: Ensure data has both `topic` (list of strings/hot-encoded) and `sentiment` (label/score).
    *   Task: Map 16 classes to Multi-label format.
*   **Action 1.2**: Data Augmentation.
    *   Task: Synthesize "short duration" or "mixed topic" samples if the dataset is unbalanced.

### Verification
*   **Check**: Run `python src/data/prepare_multitask_data.py`.
*   **Output**: Inspect `data/processed/multitask_train.csv` and ensure columns `text`, `labels` (multi-hot), `sentiment`.
*   **Test**: `pytest tests/data/test_dataset_format.py` (Create this test to verify data schema).

## Step 2: Model Architecture Upgrade (MobileBERT)
Upgrade the model to support the defined outputs.

*   **Action 2.1**: Define `MultitaskMobileBert` class in `src/models/multitask_model.py`.
    *   Backbone: `google/mobilebert-uncased` (or Chinese equivalent if applicable).
    *   Head A: `TopicClassifier` (Linear -> Sigmoid, Output Dim=16).
    *   Head B: `SentimentCombined` (Linear -> Softmax (3-class) AND Linear -> Tanh (Regression)).
*   **Action 2.2**: Implement Loss Function.
    *   `TotalLoss = w1 * BCEWithLogitsLoss(Topic) + w2 * CrossEntropyLoss(SentimentClass) + w3 * MSELoss(SentimentScore)`.

### Verification
*   **Check**: Run `python src/models/multitask_model.py` (Main block validation).
*   **Output**: Print model summary, ensure output shapes: `topic` [B, 16], `sentiment_class` [B, 3], `sentiment_score` [B, 1].
*   **Test**: `pytest tests/models/test_architecture.py`.

## Step 3: Training Loop Implementation
Implement the training pipeline.

*   **Action 3.1**: Create `src/train/train_multitask.py`.
    *   Implement training loop with `w1, w2, w3` loss weighting.
    *   Add metrics: `Micro-F1` (Topics), `Accuracy` (Sentiment), `MAE` (Score).
*   **Action 3.2**: Integration with Experiment Tracking (e.g., TensorBoard/MLFlow/Weights&Biases).

### Verification
*   **Check**: Run a "dry run" (1 epoch, small data).
    *   `python src/train/train_multitask.py --dry-run`
*   **Output**: Ensure loss decreases and logs are generated in `logs/`.
*   **Artifact**: Saved model checkpoint `checkpoints/multitask_v1.pth`.

## Step 4: CoreML Export & Quantization
Export the trained model to iOS-ready format.

*   **Action 4.1**: Update `src/export/export_multitask_coreml.py`.
    *   Convert PyTorch model to CoreML using `coremltools`.
    *   Define Inputs: `input_ids`, `attention_mask`.
    *   Define Outputs: `topic_probs`, `sentiment_probs`, `sentiment_score`.
*   **Action 4.2**: Apply Quantization (Int8 / Float16).
    *   Use `coremltools.models.neural_network.quantization_utils`.
    *   Target size: < 40MB.

### Verification
*   **Check**: Run `bash scripts/export_multitask_coreml.sh`.
*   **Output**: `models/deploy/MorningGoal_v2_Int8.mlpackage`.
*   **Test**: Run a Python inference script using `coremltools` on the exported model to verify output matches PyTorch model.
    *   `python scripts/verify_coreml_model.py --model models/deploy/MorningGoal_v2_Int8.mlpackage`

## Step 5: Heuristic Rules Engine (Python Prototype)
Develop the rules for Duration Extraction and "Implicit Sentiment" correction in Python first.

*   **Action 5.1**: Create `src/rules/duration_extractor.py` (Regex for "2 hours", "30 mins").
*   **Action 5.2**: Create `src/rules/sentiment_corrector.py` (Rule lists for negations/short sentences).

### Verification
*   **Test**: `pytest tests/rules/test_duration.py` with test cases like "I studied for 2h", "ran for 30mins".
*   **Test**: `pytest tests/rules/test_sentiment_correction.py`.
