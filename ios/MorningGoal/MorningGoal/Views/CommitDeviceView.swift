import Combine
import SwiftUI

struct CommitDeviceView: View {
    @Binding var progress: Double
    @Binding var isCommitting: Bool
    let onCommit: () -> Void

    @State private var pressStart: Date?
    @State private var isPressed: Bool = false
    @State private var didCompleteLongPress: Bool = false
    @State private var reverseStart: Date?
    @State private var reverseDuration: Double = 0
    @State private var reverseFrom: Double = 0
    private let frameTicker = Timer.publish(every: 1.0 / 60.0, on: .main, in: .common).autoconnect()

    var body: some View {
        ZStack {
            // 背景圆环
            Circle()
                .stroke(Color.Design.sunriseGold.opacity(0.2), lineWidth: 8)
                .frame(width: 140, height: 140)

            // 进度圆环（0%→100%），使用 trim 绘制并从顶部开始
            Circle()
                .trim(from: 0, to: progress)
                .stroke(Color.Design.sunriseGold, style: StrokeStyle(lineWidth: 8, lineCap: .round))
                .frame(width: 140, height: 140)
                .rotationEffect(.degrees(-90))

            // 中心按钮（仅缩放反馈，不改变配色）
            Circle()
                .fill(Color.Design.darkIndigo)
                .frame(width: 120, height: 120)
                .overlay(
                    Image(systemName: "hand.raised.fill")
                        .font(.largeTitle)
                        .foregroundColor(Color.Design.sunriseGold)
                )
                .scaleEffect(isPressed ? 0.95 : 1.0)
        }
        .onLongPressGesture(
            minimumDuration: 3.0,
            maximumDistance: 20,
            pressing: { pressing in
                if pressing {
                    if !isPressed {
                        isPressed = true
                        isCommitting = true
                        let now = Date()
                        pressStart = now
                        reverseStart = nil
                        reverseDuration = 0
                        reverseFrom = 0
                        progress = 0
                    }
                } else {
                    guard let start = pressStart else { return }
                    let elapsed = max(0, min(Date().timeIntervalSince(start), 3.0))
                    if elapsed < 3.0 {
                        didCompleteLongPress = false
                        isPressed = false
                        isCommitting = false
                        pressStart = nil
                        reverseStart = Date()
                        reverseDuration = elapsed
                        reverseFrom = min(elapsed / 3.0, 1.0)
                    }
                }
            },
            perform: {
                didCompleteLongPress = true
                isPressed = false
                isCommitting = false
                pressStart = nil
                progress = 1.0
                let generator = UIImpactFeedbackGenerator(style: .heavy)
                generator.impactOccurred()
                onCommit()
            }
        )
        .onReceive(frameTicker) { now in
            if let start = pressStart {
                let elapsed = now.timeIntervalSince(start)
                if elapsed >= 3.0 {
                    progress = 1.0
                } else {
                    progress = max(0, min(elapsed / 3.0, 1.0))
                }
            } else if let rStart = reverseStart {
                let elapsed = now.timeIntervalSince(rStart)
                if reverseDuration <= 0 {
                    progress = 0
                    reverseStart = nil
                } else {
                    let ratio = max(0, min(elapsed / reverseDuration, 1.0))
                    progress = max(reverseFrom * (1.0 - ratio), 0)
                    if ratio >= 1.0 { reverseStart = nil }
                }
            }
        }
    }
}
