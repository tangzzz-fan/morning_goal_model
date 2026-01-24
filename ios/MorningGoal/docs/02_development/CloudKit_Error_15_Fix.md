# CloudKit Error 15/2000 Fix Guide

## Error Description

```
CKError 2/1011: "Partial Failure" - Failed to modify some record zones
CKError 15/2000: "Server Rejected Request"
```

This error means Apple's CloudKit servers rejected the attempt to create the CloudKit zone because the container isn't properly initialized yet.

## Why This Happens

1. **First-time container creation** - The container `iCloud.com.tango.MorningGoal` hasn't been created on Apple's servers yet
2. **Container provisioning delay** - Apple's servers need time to provision the container (5-10 minutes)
3. **Simulator limitations** - CloudKit can be flaky in simulator vs real device
4. **iCloud account issues** - Not properly signed into iCloud

## ✅ Solutions (In Order of Likelihood)

### Solution 1: Wait and Retry (Most Common) ⏰

The container is being created automatically. Just wait:

1. **Close the app**
2. **Wait 5-10 minutes** (Apple provisions the container)
3. **Restart the app**
4. Check console logs for successful setup

**Why this works:** Apple creates the container automatically on first launch, but it takes time to provision.

### Solution 2: Reinitialize CloudKit in Xcode 🔧

Force Xcode to recreate the container configuration:

1. Open Xcode project
2. Select target "MorningGoal"
3. Go to **Signing & Capabilities** tab
4. Click **"- Capability"** next to iCloud (remove it)
5. Click **"+ Capability"** → Add "iCloud" back
6. Check ☑️ CloudKit
7. Ensure container shows: `iCloud.com.tango.MorningGoal`
8. Clean build folder: `Cmd+Shift+K`
9. Rebuild and run

**Why this works:** Xcode re-registers the container with Apple's servers.

### Solution 3: Check iCloud Account Settings 🔐

Ensure you're properly signed into iCloud:

1. Open **Settings** app on device/simulator
2. Tap your **Apple ID** at the top
3. Tap **iCloud**
4. Ensure **iCloud Drive** is ON
5. Scroll down, ensure app has iCloud permission

**Simulator specific:**
- Xcode → Window → Devices and Simulators
- Right-click simulator → Erase All Content and Settings
- After reset, sign into iCloud again

### Solution 4: Test on Real Device 📱

Simulator CloudKit can be unreliable:

1. Connect a real iOS device
2. Ensure device is signed into iCloud (Settings → iCloud)
3. Run app on device instead of simulator
4. Real devices have more reliable CloudKit connections

### Solution 5: Check Apple Developer Portal 🌐

Verify container exists in CloudKit Dashboard:

1. Go to: https://icloud.developer.apple.com/dashboard
2. Sign in with Apple Developer account
3. Look for: `iCloud.com.tango.MorningGoal`
4. If it doesn't exist:
   - It's still being created (wait 10 minutes)
   - Or entitlements don't match

### Solution 6: Disable CloudKit Temporarily 🚫

If you need the app to work immediately without sync:

1. Open `DataController.swift`
2. Find line: `private static let cloudKitEnabled = true`
3. Change to: `private static let cloudKitEnabled = false`
4. Rebuild app

**Trade-offs:**
- ✅ App works immediately
- ✅ No CloudKit errors
- ❌ No sync across devices
- ❌ Data lost on app deletion

## Verification

### When It's Working

You should see these logs:

```
✅ Core Data loaded successfully
☁️ CloudKit sync: ENABLED
🔔 CloudKit sync monitoring enabled
☁️ CloudKit Event:
  Type: Setup
  ✅ Success
```

### When It's Still Broken

You'll see:

```
❌ Error: 未能完成操作。（CKErrorDomain错误2。）
⚠️ CloudKit Partial Failure
```

## Understanding the Timeline

| Time | What Happens |
|------|-------------|
| T+0 (First Launch) | App attempts to create CloudKit zone |
| T+0 to T+5min | CloudKit returns error 15 (zone doesn't exist) |
| T+5min to T+10min | Apple provisions container in background |
| T+10min+ | Zone creation succeeds on next launch |

## Important Notes

### The App Still Works! 🎉

Even with CloudKit errors:
- ✅ App functions normally
- ✅ Data saves locally
- ✅ All features work
- ❌ Just no sync across devices (yet)

### This is Normal

CKError 15 on first launch is **expected behavior** for new CloudKit containers. It's not a bug in your code.

### Don't Panic

The error logs look scary but:
1. Apple is provisioning your container
2. It will work automatically after a few minutes
3. Your local data is safe

## Testing After Fix

1. **Add test data:**
   ```
   - Create 2-3 goal entries
   - Wait 60 seconds
   ```

2. **Check logs:**
   ```
   Look for: "📤 Data exported to iCloud"
   ```

3. **Test sync:**
   ```
   - Delete app
   - Reinstall
   - Wait 60 seconds
   - Data should restore
   ```

## Alternative: Use Local-Only Mode

If CloudKit continues to be problematic, you can run in local-only mode:

### Pros:
- No sync errors
- Faster saves
- Works offline always
- Simpler debugging

### Cons:
- No device sync
- Data lost on deletion
- No backup to iCloud

### To Enable:
Set `cloudKitEnabled = false` in `DataController.swift`

## Getting Help

If none of these solutions work:

1. **Check console logs** - Look for specific error codes
2. **Try device vs simulator** - Different behaviors
3. **Check network** - CloudKit needs internet
4. **Verify Apple Developer account** - Must be active
5. **Create support ticket** - Apple Developer Support

## Related Documentation

- `CloudKit_Sync_Guide.md` - General CloudKit sync info
- `iCloud_Integration_Design.md` - Architecture overview
- Apple: [CloudKit Error Codes](https://developer.apple.com/documentation/cloudkit/ckerror/code)
