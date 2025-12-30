# Phase 2.1: MVP iOS Integration (iOS/Swift Side)

**Goal:** Integrate the CoreML model into the iOS app, set up the local database for structured data, and display basic stats.

## Step 1: Core Data Schema Update
Update the local database to store analysis results.

*   **Action 1.1**: Open `.xcdatamodeld`.
*   **Action 1.2**: Add attributes to `Goal` entity (or create `GoalAnalysis` entity 1:1 relation).
    *   `sentimentScore` (Double)
    *   `sentimentLabel` (String)
    *   `topics` (String/JSON or Transformable)
    *   `embedding` (Binary Data - for future use)
    *   `detectedDuration` (Integer, minutes)

### Verification
*   **Check**: Build the app. Ensure no migration crashes (if dev environment, delete app and reinstall).
*   **Test**: Write a Core Data Unit Test `GoalDataTests.swift` that creates a Goal and saves/retrieves these new fields.

## Step 2: Model Import & Wrapper
Import the `.mlpackage` and create a Swift wrapper.

*   **Action 2.1**: Drag `MorningGoal_v2_Int8.mlpackage` into Xcode project.
*   **Action 2.2**: Create `Services/AI/MorningGoalPredictor.swift`.
    *   Function: `predict(text: String) async throws -> PredictionResult`
    *   Implement Tokenization (using a Swift BERT Tokenizer library or porting vocab).
*   **Action 2.3**: Implement Post-processing.
    *   Apply Sigmoid thresholding for topics.
    *   Map sentiment probabilities to labels.

### Verification
*   **Check**: Create a temporary UI button "Test Model" that prints results to console.
*   **Test**: `MorningGoalPredictorTests.swift`. Run inference on "I am happy" and encoded check expected output.

## Step 3: Inference Pipeline (The Parser)
Connect User Input -> Model -> Database.

*   **Action 3.1**: Create `Services/AI/InsightEngine.swift` (Singleton).
*   **Action 3.2**: Implement `process(goal: Goal)`.
    *   Run when user saves a goal (or Background Task).
    *   Call `MorningGoalPredictor`.
    *   Call `DurationExtractor` (Port Python regex to Swift).
    *   Update `Goal` object with results and save.

### Verification
*   **Check**: Run the app. Add a new goal: "Read books for 1 hour".
*   **Verify**: Use a DB viewer (or print log) to see if `topics` includes "Study/Reading" and `detectedDuration` is 60.

## Step 4: Basic UI Visualization (The Dashboard)
Show the MVP pie chart.

*   **Action 4.1**: Create `Views/Insights/WeeklyReportView.swift`.
*   **Action 4.2**: Fetch goals from last 7 days.
*   **Action 4.3**: Aggregate `topics` duration or count.
*   **Action 4.4**: Draw a Pie Chart (Swift Charts) showing "Time Distribution".

### Verification
*   **Check**: Run App -> Navigate to Insights Tab.
*   **Verify**: Pie chart renders correctly. Add goals with different topics and check if chart updates.

## Step 5: Background Task Setup
Optimize performance.

*   **Action 5.1**: Register `BGTaskScheduler`.
*   **Action 5.2**: Move model inference to a background operation if it takes > 200ms or for batch processing.

### Verification
*   **Test**: Simulate Background Fetch in Xcode Debug menu. Check logs to ensure inference triggers.
