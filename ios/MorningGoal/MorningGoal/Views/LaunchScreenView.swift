import SwiftUI

struct LaunchScreenView: View {
    @State private var opacity: Double = 0
    @State private var scale: CGFloat = 0.8
    @State private var showSparkles: Bool = false

    var body: some View {
        ZStack {
            // 深靛蓝背景 - 符合设计系统
            Color.Design.deepIndigo
                .ignoresSafeArea()

            VStack(spacing: Spacing.xl) {
                Spacer()

                // 主图标/标志
                ZStack {
                    // 背景光晕
                    Circle()
                        .fill(
                            RadialGradient(
                                colors: [
                                    Color.Design.sunriseGold.opacity(0.3),
                                    Color.Design.sunriseGold.opacity(0.0)
                                ],
                                center: .center,
                                startRadius: 0,
                                endRadius: 80
                            )
                        )
                        .frame(width: 160, height: 160)
                        .opacity(opacity)

                    // 主图标 - 日出/目标的象征
                    Image(systemName: "sun.horizon.fill")
                        .font(.system(size: 64))
                        .foregroundStyle(
                            LinearGradient(
                                colors: [Color.Design.sunriseGold, Color.Design.sunriseGold.opacity(0.8)],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )
                        .opacity(opacity)
                        .scaleEffect(scale)

                    // 闪光效果
                    if showSparkles {
                        ForEach(0 ..< 8, id: \.self) { index in
                            Image(systemName: "sparkle")
                                .font(.system(size: 12))
                                .foregroundColor(Color.Design.sunriseGold)
                                .offset(
                                    x: cos(Double(index) * .pi / 4) * 70,
                                    y: sin(Double(index) * .pi / 4) * 70
                                )
                                .opacity(opacity * 0.8)
                                .scaleEffect(0.5)
                        }
                    }
                }

                // 应用名称
                Text(LocalizedStringKey("app_name"))
                    .font(.system(size: 32, weight: .light, design: .rounded))
                    .foregroundColor(Color.Design.softWhite)
                    .opacity(opacity)

                // 标语
                Text(LocalizedStringKey("app_tagline"))
                    .font(.system(size: 14))
                    .foregroundColor(Color.Design.mutedGray)
                    .opacity(opacity * 0.8)

                Spacer()
                Spacer()
            }
        }
        .onAppear {
            // 启动动画序列
            withAnimation(.easeOut(duration: 0.8)) {
                opacity = 1.0
                scale = 1.0
            }

            // 延迟显示闪光效果
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
                withAnimation(.easeInOut(duration: 0.6)) {
                    showSparkles = true
                }
            }
        }
    }
}

#Preview {
    LaunchScreenView()
}
