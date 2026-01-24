# CloudKit Sync Guide

## Overview

This guide explains how CloudKit sync works in MorningGoal and answers common questions about data persistence across app reinstalls.

## How CloudKit Sync Works

### Automatic Synchronization

`NSPersistentCloudKitContainer` automatically syncs your Core Data to iCloud:

1. **Export (Upload)**: Local changes → iCloud (happens within seconds)
2. **Import (Download)**: iCloud → Local device (happens on app launch and periodically)
3. **Background Sync**: Continues even when app is in background (via remote notifications)

### Sync Timeline

| Event | Time Frame | Notes |
|-------|------------|-------|
| Save to local Core Data | Instant | Data saved immediately on device |
| Upload to iCloud | 5-30 seconds | Depends on network speed |
| Download after reinstall | 10-60 seconds | Initial sync after fresh install |
| Background sync | Minutes | Periodic updates from iCloud |

## ⚠️ Important: Data Recovery After Deletion

### Why Data Might Not Appear Immediately

When you delete and reinstall the app:

1. **Local database is erased** - This is expected
2. **iCloud still has your data** - Safely stored in CloudKit
3. **Initial sync takes time** - 10-60 seconds to download

### Best Practices

#### Before Deleting App (Testing)

1. **Create some data** - Add at least 2-3 goal entries
2. **Wait 30-60 seconds** - Keep app in foreground
3. **Check console logs** - Look for "📤 Data exported to iCloud"
4. **Then delete app** - Now safe to uninstall

#### After Reinstalling App

1. **Launch the app** - Opens with empty database initially
2. **Keep app in foreground** - Stay on app for 30-60 seconds
3. **Watch console logs** - Look for "📥 Data imported from iCloud"
4. **Data should appear** - Entries will populate automatically

### Checking Sync Status

We've added a CloudKit sync status view:

1. Open the app in DEBUG mode
2. Tap the debug menu button (⚙️ gear icon)
3. Select "☁️ CloudKit 同步状态"
4. View:
   - Sync status (syncing/complete)
   - Last sync timestamp
   - Local entry count
   - Any sync errors

## Implementation Details

### DataController Enhancements

```swift
// DataController now conforms to ObservableObject
final class DataController: ObservableObject {
    @Published var isSyncing: Bool = false
    @Published var lastSyncDate: Date?
    @Published var syncError: Error?
    
    // Monitors CloudKit events
    private func setupCloudKitNotifications() { ... }
    
    // Logs detailed sync status
    func checkCloudKitStatus() { ... }
}
```

### Console Logging

The app now logs detailed CloudKit events:

```
🔔 CloudKit sync monitoring enabled
✅ Core Data loaded successfully

☁️ CloudKit Event:
  Type: Export
  Start: 2026-01-19 10:23:45
  End: 2026-01-19 10:23:46
  ✅ Success
  📤 Data exported to iCloud

☁️ CloudKit Event:
  Type: Import
  Start: 2026-01-19 10:24:15
  End: 2026-01-19 10:24:16
  ✅ Success
  📥 Data imported from iCloud

📊 CloudKit Status Check:
  Last sync: 2026-01-19 10:24:16
  Is syncing: false
  No errors
  Local entries: 5
```

## Troubleshooting

### Problem: No Data After Reinstall

**Possible Causes:**

1. **Didn't wait long enough before deletion**
   - Solution: Keep app open 60s after creating data

2. **Didn't wait long enough after reinstall**
   - Solution: Keep app open 60s after launching

3. **Not signed into iCloud**
   - Solution: Settings → Sign in to iCloud

4. **Network issues**
   - Solution: Check internet connection

5. **CloudKit container not configured**
   - Solution: Check Developer Portal for container

### Problem: Data Syncing Slowly

**Normal Behavior:**
- First sync: 10-60 seconds
- Subsequent syncs: 5-30 seconds
- Large data sets: Up to 2-3 minutes

**If Slower:**
- Check network speed
- Check iCloud storage (Settings → iCloud)
- Restart device

### Problem: Duplicate Entries

**Why It Happens:**
- CloudKit doesn't support unique constraints
- App handles uniqueness at application level

**Prevention:**
- `TodayInputView` checks for existing entry before saving
- Merge policy: `NSMergeByPropertyObjectTrumpMergePolicy`

## CloudKit Requirements

### Entitlements (Already Configured)

```xml
<key>com.apple.developer.icloud-container-identifiers</key>
<array>
    <string>iCloud.com.tango.MorningGoal</string>
</array>
<key>com.apple.developer.icloud-services</key>
<array>
    <string>CloudKit</string>
</array>
```

### Info.plist (Already Configured)

```xml
<key>UIBackgroundModes</key>
<array>
    <string>remote-notification</string>
</array>
```

### Core Data Model Requirements

✅ **Fixed Issues:**

1. ~~Non-optional attributes need default values~~ ✅ Added
2. ~~Unique constraints not supported~~ ✅ Removed from schema
3. ~~Missing sync monitoring~~ ✅ Implemented

## Testing CloudKit Sync

### Manual Test Procedure

1. **Setup Phase:**
   ```
   - Clean install app
   - Sign into iCloud (Settings app)
   - Launch MorningGoal
   ```

2. **Create Data:**
   ```
   - Add 3 goal entries on different days
   - Wait 60 seconds (check console for export logs)
   - Open debug menu → Check CloudKit status
   - Verify "Last sync" shows recent timestamp
   ```

3. **Delete & Restore:**
   ```
   - Delete app from device
   - Reinstall app
   - Launch app, keep in foreground
   - Wait 60 seconds
   - Check console for import logs
   - Verify all 3 entries appear
   ```

### Automated Testing

See `DataControllerTests.swift` for Core Data tests. CloudKit sync is difficult to test automatically because:
- Requires real iCloud account
- Network-dependent
- Time-dependent

## FAQ

### Q: Do I need to keep the app open for sync?

**A:** No, but it helps:
- **Background sync works** - Via remote notifications
- **Foreground is faster** - More aggressive sync scheduling
- **Initial sync**: Keep open 30-60s after fresh install

### Q: Will my data be lost if I delete the app?

**A:** No, as long as:
- You waited 30-60s after creating data (for upload)
- You're signed into same iCloud account
- CloudKit container is properly configured

### Q: Can I force an immediate sync?

**A:** Not directly. CloudKit manages sync automatically. But you can:
- Save data (`DataController.shared.save()`)
- Keep app in foreground
- Ensure good network connection

### Q: How much iCloud storage does this use?

**A:** Minimal:
- Each goal entry: ~500 bytes
- 1000 entries: ~500 KB
- Years of data: < 5 MB

### Q: Can other people see my data?

**A:** No:
- Data stored in **private** CloudKit database
- Only accessible with your iCloud account
- Developer cannot access your data

## References

- [Apple: NSPersistentCloudKitContainer](https://developer.apple.com/documentation/coredata/nspersistentcloudkitcontainer)
- [Apple: Setting Up Core Data with CloudKit](https://developer.apple.com/documentation/coredata/mirroring_a_core_data_store_with_cloudkit)
- Project: `iCloud_Integration_Design.md`
