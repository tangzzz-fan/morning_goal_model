// 示例: 如何在应用中集成模型调试界面

import SwiftUI

// 方法 1: 在设置页面中添加
struct SettingsView: View {
    #if DEBUG
    @State private var showModelDebug = false
    #endif
    
    var body: some View {
        List {
            // ... 其他设置项
            
            #if DEBUG
            Section("开发者选项") {
                Button(action: { showModelDebug = true }) {
                    Label("模型调试", systemImage: "cpu")
                }
            }
            #endif
        }
        #if DEBUG
        .sheet(isPresented: $showModelDebug) {
            ModelDebugView()
        }
        #endif
    }
}

// 方法 2: 在主界面添加隐藏入口 (摇一摇)
struct ContentView: View {
    #if DEBUG
    @State private var showModelDebug = false
    #endif
    
    var body: some View {
        YourMainView()
            #if DEBUG
            .onShake {
                showModelDebug = true
            }
            .sheet(isPresented: $showModelDebug) {
                ModelDebugView()
            }
            #endif
    }
}

// 摇一摇检测扩展
#if DEBUG
extension View {
    func onShake(perform action: @escaping () -> Void) -> some View {
        self.modifier(ShakeModifier(action: action))
    }
}

struct ShakeModifier: ViewModifier {
    let action: () -> Void
    
    func body(content: Content) -> some View {
        content
            .onReceive(NotificationCenter.default.publisher(for: UIDevice.deviceDidShakeNotification)) { _ in
                action()
            }
    }
}

extension UIDevice {
    static let deviceDidShakeNotification = Notification.Name(rawValue: "deviceDidShakeNotification")
}

extension UIWindow {
    open override func motionEnded(_ motion: UIEvent.EventSubtype, with event: UIEvent?) {
        if motion == .motionShake {
            NotificationCenter.default.post(name: UIDevice.deviceDidShakeNotification, object: nil)
        }
    }
}
#endif

// 方法 3: 在导航栏添加按钮
struct MainView: View {
    #if DEBUG
    @State private var showModelDebug = false
    #endif
    
    var body: some View {
        NavigationView {
            YourContentView()
                .navigationTitle("Morning Goal")
                #if DEBUG
                .toolbar {
                    ToolbarItem(placement: .navigationBarTrailing) {
                        Button(action: { showModelDebug = true }) {
                            Image(systemName: "cpu")
                        }
                    }
                }
                .sheet(isPresented: $showModelDebug) {
                    ModelDebugView()
                }
                #endif
        }
    }
}

// 方法 4: 使用长按手势
struct AnyView: View {
    #if DEBUG
    @State private var showModelDebug = false
    @State private var pressCount = 0
    #endif
    
    var body: some View {
        YourView()
            #if DEBUG
            .onTapGesture(count: 3) {
                // 三连击打开调试界面
                showModelDebug = true
            }
            .sheet(isPresented: $showModelDebug) {
                ModelDebugView()
            }
            #endif
    }
}
