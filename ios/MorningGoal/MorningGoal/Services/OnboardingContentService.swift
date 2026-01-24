import Combine
import Foundation

/// Onboarding 内容服务
/// 负责提供 Onboarding 流程中的文案内容，支持随机文案组和未来的 ML 生成扩展
class OnboardingContentService: ObservableObject {
    static let shared = OnboardingContentService()

    private var cachedSlides: [FeatureSlideContent] = []

    private init() {
        // 初始化时生成一次随机文案，确保整个会话期间一致
        cachedSlides = generateRandomSlides()
    }

    /// 获取功能轮播的内容列表
    /// - Returns: 功能轮播页面数据列表
    func getFeatureSlides() -> [FeatureSlideContent] {
        return cachedSlides
    }

    private func generateRandomSlides() -> [FeatureSlideContent] {
        // 使用新定义的 3 个核心页面文案
        // 未来可以添加 variants
        return [
            FeatureSlideContent(
                id: "theory",
                title: NSLocalizedString("onboarding_page1_title", comment: ""),
                subtitle: NSLocalizedString("onboarding_page1_subtitle", comment: ""),
                imageSystemName: "chart.xyaxis.line", // 象征 42% 提升
                colorName: "sunriseGold"
            ),
            FeatureSlideContent(
                id: "focus",
                title: NSLocalizedString("onboarding_page2_title", comment: ""),
                subtitle: NSLocalizedString("onboarding_page2_subtitle", comment: ""),
                imageSystemName: "target",
                colorName: "deepIndigo"
            ),
            FeatureSlideContent(
                id: "growth",
                title: NSLocalizedString("onboarding_page3_title", comment: ""),
                subtitle: NSLocalizedString("onboarding_page3_subtitle", comment: ""),
                imageSystemName: "flame.fill",
                colorName: "orange"
            )
        ]
    }

    /// 获取欢迎页的随机文案
    /// - Returns: 欢迎标题和副标题
    func getWelcomeMessage() -> (title: String, subtitle: String) {
        let options = [
            (
                title: NSLocalizedString("onboarding_welcome_title_1", comment: ""),
                subtitle: NSLocalizedString("onboarding_welcome_subtitle_1", comment: "")
            ),
            (
                title: NSLocalizedString("onboarding_welcome_title_2", comment: ""),
                subtitle: NSLocalizedString("onboarding_welcome_subtitle_2", comment: "")
            )
        ]
        return options.randomElement() ?? options[0]
    }

    /// [扩展接口] 未来用于获取 ML 生成的个性化文案
    /// - Parameter userContext: 用户上下文信息（如来源、设备信息等）
    /// - Returns: 个性化文案
    func fetchMLGeneratedContent(userContext: [String: Any]) async -> [FeatureSlideContent]? {
        // 预留接口：
        // 1. 发送请求到后端或调用本地 ML 模型
        // 2. 根据用户画像生成针对性的文案
        // 3. 返回生成的内容
        return nil
    }
}

/// 功能轮播页面内容模型
struct FeatureSlideContent: Identifiable {
    let id: String
    let title: String
    let subtitle: String
    let imageSystemName: String
    let colorName: String
}
