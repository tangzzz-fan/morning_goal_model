# MobileBERT On-Device Training Integration Guide

## 1. Overview
This document outlines the steps to integrate the new `student_sequence_classification.mlpackage` into the Morning Goal app for on-device training (Personalization).

**Current Status:**
- **Model**: `MorningGoal/coreml/student_sequence_classification.mlpackage` exists.
- **Infrastructure**:
    - `ModelUpdateScheduler` is responsible for triggering updates.
    - `GoalEntry` tracks user corrections (`isTrainingSample`).
    - `AdaptiveModelService` exists but points to an old model ("GoalClassifier") and `updateModel` is unimplemented.
- **Objective**: enable the app to fine-tune the classification model based on user corrections.

## 2. Prerequisites
- **Updatable Model**: Ensure `student_sequence_classification.mlpackage` is marked as "Updatable" in Xcode CoreML editor.
    - **Trainable Layers**: Typically the final Dense layer (classifier head).
    - **Loss Function**: Cross Entropy Loss.
    - **Optimizer**: SGD or Adam (configure default hyperparameters).
    - **Inputs**: `input_ids`, `attention_mask` (Int32, Shape 1x128).
    - **Target**: `target` (String or Int, matching category labels).

## 3. Development Steps

### Step 1: Update Model Loading Logic
Modify `AdaptiveModelService.swift` to load the correct model and handle first-run vs. updated model loading.

```swift
// Pseudo-code for init
let fileManager = FileManager.default
let docURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
let updatedModelURL = docURL.appendingPathComponent("student_sequence_classification.mlmodelc")

if fileManager.fileExists(atPath: updatedModelURL.path) {
    // Load updated model
    self.model = try MLModel(contentsOf: updatedModelURL)
} else {
    // Load bundled model
    let bundleURL = Bundle.main.url(forResource: "student_sequence_classification", withExtension: "mlpackage")!
    // Must compile first if it's mlpackage, or load directly if compiled
    self.model = try MLModel(contentsOf: bundleURL)
}
```

### Step 2: Implement Data Provider
Create a `MLBatchProvider` from `[TrainingSample]`.

- Use `BertTokenizer` to tokenize `TrainingSample.text`.
- Convert tokens to `input_ids` and `attention_mask` `MLMultiArray`.
- Convert `TrainingSample.correctCategory` to target label (ensure label mapping matches model output classes).
- **Critical**: Ensure tokenization logic matches the training-time tokenization exactly.

### Step 3: Implement `updateModel`
Implement the actual training task using `MLUpdateTask`.

```swift
func updateModel(with samples: [TrainingSample]) async throws {
    // 1. Prepare Training Data
    let trainingData = try prepareBatchProvider(from: samples)
    
    // 2. Load Updatable Model URL
    // access the compile model path
    
    // 3. Create Update Task
    let updateTask = try MLUpdateTask(forModelAt: modelURL, trainingData: trainingData, configuration: config)
    
    // 4. Resume and Await
    // Handle completion handler to save the new model
}
```

### Step 4: Handle Model Persistence
- On successful training, save the *updated* model to the Documents directory.
- Reload `self.model` with the newly trained model so immediate inference uses it.

### Step 5: Safety Checks
- **Background Time**: Ensure training is fast enough or use background processing task time efficiently.
- **Rollback**: If training diverges (accuracy drops), potentially discard the update. *Note: validating on a small validation set (e.g. recent correct entries) is recommended.*

## 4. Verification
1.  **Manual Trigger**: Use `ModelUpdateScheduler.triggerUpdateImmediately()` to test.
2.  **Logs**: Observe `AdaptiveModelService` logs for "update_success" or loss values.
3.  **Effect**: Correct a goal category, run update, and verify the same goal is classified correctly afterwards.
