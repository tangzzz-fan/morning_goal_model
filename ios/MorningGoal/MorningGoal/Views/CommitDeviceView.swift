import Combine
import SwiftUI

struct CommitDeviceView: View {
    @Binding var progress: Double
    @Binding var isCommitting: Bool
    let size: CGFloat
    let duration: TimeInterval
    let onCommit: () -> Void

    // Default init for backward compatibility or ease of use
    init(progress: Binding<Double>, isCommitting: Binding<Bool>, size: CGFloat = 140, duration: TimeInterval = 3.0, onCommit: @escaping () -> Void) {
        self._progress = progress
        self._isCommitting = isCommitting
        self.size = size
        self.duration = duration
        self.onCommit = onCommit
    }

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
                .stroke(Color.Design.sunriseGold.opacity(0.2), lineWidth: size * 0.06) // Scale line width
                .frame(width: size, height: size)

            // 进度圆环（0%→100%），使用 trim 绘制并从顶部开始
            Circle()
                .trim(from: 0, to: progress)
                .stroke(Color.Design.sunriseGold, style: StrokeStyle(lineWidth: size * 0.06, lineCap: .round))
                .frame(width: size, height: size)
                .rotationEffect(.degrees(-90))

            // 中心按钮（仅缩放反馈，不改变配色）
            Circle()
                .fill(Color.Design.darkIndigo)
                .frame(width: size * 0.85, height: size * 0.85)
                .overlay(
                    Image(systemName: "hand.raised.fill")
                        .font(.system(size: size * 0.3)) // Scale font
                        .foregroundColor(Color.Design.sunriseGold)
                )
                .scaleEffect(isPressed ? 0.95 : 1.0)
        }
        .onLongPressGesture(
            minimumDuration: duration,
            maximumDistance: 50, // Increased tolerance
            pressing: { pressing in
                if pressing {
                    if !isPressed {
                        isPressed = true
                        isCommitting = true
                        let now = Date()

                        // If reversing, calculate effective start time to resume smoothly
                        if let rStart = reverseStart {
                            let elapsedReverse = now.timeIntervalSince(rStart)
                            // Current visual progress
                            let currentP = max(0, min((reverseDuration > 0 ? (1.0 - elapsedReverse / reverseDuration) : 0) * reverseFrom, 1.0))

                            // To match this progress, we pretend we started (currentP * duration) seconds ago
                            let effectiveElapsed = currentP * duration
                            pressStart = now.addingTimeInterval(-effectiveElapsed)
                        } else {
                            pressStart = now
                        }

                        reverseStart = nil
                        // Do not reset progress immediately to avoid visual jump, let timer catch up
                    }
                } else {
                    guard let start = pressStart else { return }
                    let elapsed = max(0, min(Date().timeIntervalSince(start), duration))
                    if elapsed < duration {
                        didCompleteLongPress = false
                        isPressed = false
                        isCommitting = false
                        pressStart = nil
                        reverseStart = Date()
                        reverseDuration = elapsed // Reverse takes same time as press duration so far? Or fixed speed?
                        // Let's make reverse slightly faster for better feel: 0.5x duration or just elapsed
                        reverseFrom = min(elapsed / duration, 1.0)

                        // Current progress is:
                        progress = reverseFrom
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
                if elapsed >= duration {
                    progress = 1.0
                } else {
                    progress = max(0, min(elapsed / duration, 1.0))
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
