# Phase 2.3: Pro Features (Advanced Insights)

**Goal:** Add "Personal Coach" capabilities: Burnout Prediction, Habit Associations (Context Awareness), and refined Sentiment.

## Step 1: Sentiment Regression & Refinement
Replace classification with fine-grained scoring.

*   **Action 1.1**: Update iOS Model Wrapper to use the `sentiment_score` output (Float -1 to 1).
*   **Action 1.2**: Implement `SentimentSmoother`.
    *   Apply Rules: If valid text length < 5, dampen the score.
    *   Apply Dictionary Check: If "tired", "exhausted" present, force score lower.
    *   Apply Energy Level mapping.

### Verification
*   **Test**: `SentimentLogicTests.swift`.
    *   Input: "I am okay." (Model: 0.1) -> Output: 0.1.
    *   Input: "Exhausted." (Model: -0.2) -> Rule Correction -> -0.8.

## Step 2: Burnout Alert System (Trend Analysis)
Detect negative trends over time.

*   **Action 2.1**: Create `Analysis/TrendAnalyzer.swift`.
*   **Action 2.2**: Algorithm:
    *   Fetch last 7 days `energyLevel`.
    *   Calculate Slope (Linear Regression).
    *   If Slope < Threshold AND Avg < Threshold -> Trigger Alert.
*   **Action 2.3**: Persistence check (don't spam alerts).

### Verification
*   **Test**: `TrendAnalyzerTests.swift`. Feed mock data (decreasing energy). Verify `isBurnoutRisk` returns true.

## Step 3: Insight Cards & Smart Tips
The "Coach" UI.

*   **Action 3.1**: Create `SmartTipEngine`.
*   **Action 3.2**: Define Tip Templates.
    *   "You seem tired..."
    *   "You are on a streak..."
*   **Action 3.3**: Create `SmartTipView` on Dashboard.

### Verification
*   **Check**: Manually manipulate DB data to trigger rules. Check if Cards appear in UI.

## Step 4: Optimization & Federated Analytics
Final polish.

*   **Action 4.1**: Implement "Telemetry" (Private).
    *   Record correction rate (Did user change the topic?).
*   **Action 4.2**: Performance profiling.
    *   Instruments: Check Memory usage of `InsightEngine`.
    *   Verify < 40MB RAM spike.

### Verification
*   **Check**: Profile in Xcode Instruments for 10 minutes of usage.
