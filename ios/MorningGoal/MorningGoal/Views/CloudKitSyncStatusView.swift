//
//  CloudKitSyncStatusView.swift
//  MorningGoal
//
//  Created for CloudKit sync monitoring
//

import CoreData
import SwiftUI

struct CloudKitSyncStatusView: View {
    @ObservedObject private var dataController = DataController.shared
    @Environment(\.managedObjectContext) private var context

    @State private var entryCount: Int = 0
    @State private var isRefreshing = false

    var body: some View {
        NavigationView {
            List {
                Section("Sync Status") {
                    HStack {
                        Text("Is Syncing")
                        Spacer()
                        if dataController.isSyncing {
                            ProgressView()
                        } else {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                        }
                    }

                    HStack {
                        Text("Last Sync")
                        Spacer()
                        Text(dataController.lastSyncDate?.formatted() ?? "Never")
                            .foregroundColor(.secondary)
                    }

                    if let error = dataController.syncError {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Last Error")
                                .font(.headline)
                            Text(error.localizedDescription)
                                .font(.caption)
                                .foregroundColor(.red)
                        }
                    }
                }

                Section("Local Data") {
                    HStack {
                        Text("Goal Entries")
                        Spacer()
                        Text("\(entryCount)")
                            .foregroundColor(.secondary)
                    }
                }

                Section("Actions") {
                    Button(action: refreshStatus) {
                        HStack {
                            Text("Refresh Status")
                            Spacer()
                            if isRefreshing {
                                ProgressView()
                            }
                        }
                    }

                    Button(action: checkStatus) {
                        Text("Check CloudKit Status (Console)")
                    }
                }

                Section("Tips") {
                    VStack(alignment: .leading, spacing: 8) {
                        if let error = dataController.syncError {
                            let nsError = error as NSError
                            if nsError.domain == "CKErrorDomain" && (nsError.code == 2 || nsError.code == 15) {
                                Text("⚠️ CloudKit Container Not Ready:")
                                    .font(.headline)
                                    .foregroundColor(.orange)

                                Text("The CloudKit container needs initialization. Try:")
                                    .font(.caption)
                                    .padding(.top, 2)

                                Text("1️⃣ Wait 5-10 minutes, then restart app")
                                Text("2️⃣ Check iCloud login: Settings → iCloud")
                                Text("3️⃣ Xcode → Signing & Capabilities → Toggle iCloud off/on")
                                Text("4️⃣ Try on a real device (not simulator)")
                                    .padding(.bottom, 4)
                            } else {
                                Text("❌ Sync Error Detected")
                                    .font(.headline)
                                    .foregroundColor(.red)
                                Text("Check console logs for details")
                                    .font(.caption)
                            }
                        } else {
                            Text("💡 CloudKit Sync Tips:")
                                .font(.headline)

                            Text("• After fresh install, keep app open for 30-60 seconds")
                            Text("• Data syncs automatically in background")
                            Text("• Check Console/Debug output for detailed sync logs")
                            Text("• Ensure you're signed into iCloud on this device")
                        }
                    }
                    .font(.caption)
                }
            }
            .navigationTitle("CloudKit Sync")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear(perform: refreshStatus)
        }
    }

    private func refreshStatus() {
        isRefreshing = true

        let request = NSFetchRequest<NSFetchRequestResult>(entityName: "GoalEntry")
        request.includesSubentities = false

        do {
            entryCount = try context.count(for: request)
        } catch {
            debugPrint("Failed to count entries: \(error)")
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            isRefreshing = false
        }
    }

    private func checkStatus() {
        dataController.checkCloudKitStatus()
    }
}

#Preview {
    CloudKitSyncStatusView()
        .environment(\.managedObjectContext, DataController.shared.container.viewContext)
}
