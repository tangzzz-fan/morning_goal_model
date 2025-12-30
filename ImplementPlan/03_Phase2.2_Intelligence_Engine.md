# Phase 2.2: Intelligence Engine (Embeddings & Clustering)

**Goal:** Implement vector storage and K-Means clustering to discover hidden life themes ("Focus Bubbles").

## Step 1: Vector Storage Mechanism
Implement a way to store and retrieve vectors efficiently (Simulation of Vector DB).

*   **Action 1.1**: Define `VectorStore` class in Swift.
    *   Option A: Store as binary files on disk (`vectors.bin`).
    *   Option B: Store in Core Data `Binary Data` field (if < 10k items).
    *   Decision: Start with Core Data `embedding` field for MVP.
*   **Action 1.2**: Implement `fetchVectors(dateRange: DateInterval) -> [[Float]]`.

### Verification
*   **Test**: `VectorStoreTests.swift`. Save 100 vectors, fetch them back, check precision.

## Step 2: On-Device K-Means Clustering
Implement or import a lightweight clustering algorithm.

*   **Action 2.1**: Implement `KMeans` class in Swift (Accelerate framework / vDSP).
    *   Input: `[[Float]]` (Vectors).
    *   Output: `centroids`, `cluster_assignments`.
    *   Parameters: `k` (dynamic or fixed, e.g., 5).
*   **Action 2.2**: Integrate `ClusterService`.
    *   Run weekly or on-demand.
    *   Assign each Goal to a Cluster ID.

### Verification
*   **Test**: `KMeansTests.swift`. Create synthetic 2D data (clearly separated groups). Run KMeans and verify it separates them correctly.

## Step 3: Keyword Extraction (for Bubbles)
Label the clusters.

*   **Action 3.1**: For each cluster, aggregate the text of goals.
*   **Action 3.2**: Implement `TF-IDF` or simple Frequency Counter (negating stop words) to find representative words for the cluster.
*   **Action 3.3**: Generate "Card Title" for the cluster (e.g., "Reading", "Project X").

### Verification
*   **Check**: Run on real data. Check if "Cluster 1" (which contains "run", "gym", "swim") gets the label "run" or "gym".

## Step 4: Dashboard Upgrade - Focus Bubbles
Visualizing the clusters.

*   **Action 4.1**: Create `Views/Insights/Components/BubbleChart.swift`.
*   **Action 4.2**: Render bubbles where Size = Count/Duration, Color = Sediment.
*   **Action 4.3**: Integrate into `WeeklyReportView`.

### Verification
*   **Check**: UI Inspection. Visual check of bubble physics and rendering.
