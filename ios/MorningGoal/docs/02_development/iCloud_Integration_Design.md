# iCloud Storage Integration Design Document

## 1. Overview
**Objective**: Implement data persistence across app reinstalls and device synchronization using iCloud.
**Current State**: Local-only storage using `NSPersistentContainer` (Core Data). Data is lost upon app deletion.
**Target State**: Synced storage using `NSPersistentCloudKitContainer`. Data persists in iCloud and syncs across user devices.

## 2. Architecture Changes

### 2.1 Persistence Controller (`DataController`)
- **Transition**: Replace `NSPersistentContainer` with `NSPersistentCloudKitContainer`.
- **Reasoning**: `NSPersistentCloudKitContainer` provides a seamless wrapper around Core Data that automatically mirrors data to a private CloudKit database. It handles networking, caching, and synchronization logic automatically.

### 2.2 Data Model (`MorningGoalModel`)
- **Entities**: `GoalEntry`, `UserSettings`.
- **CloudKit Requirements**:
    - CloudKit generally prefers optional attributes to handle "eventual consistency" where a record might arrive partially. However, our current strict model (`optional: false`) is acceptable **IF** we ensure valid defaults are always populated before saving.
    - **Constraint**: `GoalEntry` uses `dateString` as a uniqueness constraint. `NSPersistentCloudKitContainer` supports this but conflict resolution policies (`NSMergeByPropertyObjectTrumpMergePolicy`) must be strictly enforced (already present in `DataController`).

## 3. Migration & Sync Strategy

### 3.1 Local to Cloud Migration
- When the app updates to the new `NSPersistentCloudKitContainer`:
    - The existing local `.sqlite` file can be used as the store.
    - CloudKit will automatically upload existing local records to iCloud upon first launch (if the store description is configured correctly to match the existing store URL).
    - **Risk**: If the store URL changes, the app will create a *new* database and the user will look like they lost data until they sign in (or data won't merge). We must ensure the `persistentStoreDescriptions` points to the existing local store location.

### 3.2 Synchronization Policy
- **Merge Policy**: `NSMergeByPropertyObjectTrumpMergePolicy`. In case of conflict (e.g., editing the same day on two devices), the latest change (in-memory/current device) usually wins or is merged by property.
- **UI Updates**: The `viewContext` is already configured with `automaticallyMergesChangesFromParent = true`. This ensures SwiftUI views using `@FetchRequest` will update automatically when remote changes arrive.

## 4. Implementation Steps

1.  **Entitlements Configuration**:
    - Add `com.apple.developer.icloud-container-identifiers`.
    - Add `com.apple.developer.icloud-services` (CloudKit).
    - Enable "Background Modes": Remote notifications (required for silent sync updates).

2.  **Code Modifications (`DataController.swift`)**:
    - Change container type.
    - Configure `NSPersistentStoreDescription` to enable history tracking and remote notifications (crucial for robust sync).

3.  **Verification**:
    - Uninstall/Reinstall test: Ensure data restores.
    - Multi-device test: Ensure data syncs in near real-time.

## 5. Fallback & Privacy
- **Offline Mode**: The app continues to work 100% offline. Sync happens when connectivity is restored.
- **Privacy**: Data is stored in the user's private iCloud database. The developer has no access to this data.

## 6. Future Considerations (v2.0 AI)
- Since v2.0 involves on-device AI training, we must ensure that **training data is not implicitly uploaded** if it's large or sensitive, or if we want to keep training local.
- Current AI attributes (`sentiment`, `category`) are lightweight strings/doubles, which is fine for CloudKit.
- If we store large embeddings later, we might need a separate "Local Only" configuration for those specific entities/attributes.
