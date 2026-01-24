import StoreKit
import SwiftUI

struct PaywallView: View {
    @State private var storeManager = StoreManager.shared
    @Environment(\.dismiss) private var dismiss
    let onPurchaseSuccess: () -> Void

    var body: some View {
        ZStack {
            Color.Design.deepIndigo.ignoresSafeArea()

            VStack(spacing: Spacing.lg) {
                // Header
                VStack(spacing: Spacing.md) {
                    Text("Unlock Your Full Potential")
                        .font(Typography.title)
                        .foregroundColor(Color.Design.softWhite)
                        .multilineTextAlignment(.center)

                    Text("Keep your habit going. Unlock unlimited history and future AI insights.")
                        .font(Typography.body)
                        .foregroundColor(Color.Design.mutedGray)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, Spacing.xl)

                // Features List
                VStack(alignment: .leading, spacing: Spacing.md) {
                    FeatureRow(icon: "clock.arrow.circlepath", text: "Unlimited History Access")
                    FeatureRow(icon: "lock.shield.fill", text: "100% Private & Offline")
                    FeatureRow(icon: "sparkles", text: "Future AI Insights (Coming Soon)")
                    FeatureRow(icon: "heart.fill", text: "Support Independent Dev")
                }
                .padding(.vertical, Spacing.lg)

                Spacer()

                // Products
                if storeManager.isLoading {
                    ProgressView()
                        .tint(Color.Design.sunriseGold)
                } else if let error = storeManager.errorMessage {
                    Text(error)
                        .foregroundColor(.red)
                        .font(Typography.caption)
                    Button("Retry") {
                        Task { await storeManager.loadProducts() }
                    }
                } else {
                    VStack(spacing: Spacing.md) {
                        ForEach(storeManager.products) { product in
                            ProductButton(product: product) {
                                Task {
                                    do {
                                        try await storeManager.purchase(product)
                                        onPurchaseSuccess()
                                    } catch {
                                        print("Purchase failed: \(error)")
                                    }
                                }
                            }
                        }
                    }
                }

                // Restore & Legal
                VStack(spacing: Spacing.sm) {
                    Button("Restore Purchases") {
                        Task { await storeManager.restorePurchases() }
                    }
                    .font(Typography.caption)
                    .foregroundColor(Color.Design.sunriseGold)

                    HStack(spacing: Spacing.md) {
                        if let termsURL = URL(string: "https://mcnrn1su375m.feishu.cn/wiki/MX9Nw1u5uiFuaSk1aeCcI0Slnrd") {
                            Link("Terms of Service", destination: termsURL)
                        } else {
                            Text("Terms of Service")
                        }
                        if let privacyURL = URL(string: "https://mcnrn1su375m.feishu.cn/wiki/MX9Nw1u5uiFuaSk1aeCcI0Slnrd") {
                            Link("Privacy Policy", destination: privacyURL)
                        } else {
                            Text("Privacy Policy")
                        }
                    }
                    .font(.system(size: 10))
                    .foregroundColor(Color.Design.mutedGray)
                }
                .padding(.bottom, Spacing.lg)
            }
            .padding(.horizontal, Spacing.xl)
        }
        .onAppear {
            Task { await storeManager.loadProducts() }
        }
    }
}

struct FeatureRow: View {
    let icon: String
    let text: String

    var body: some View {
        HStack(spacing: Spacing.md) {
            Image(systemName: icon)
                .foregroundColor(Color.Design.sunriseGold)
                .frame(width: 24)
            Text(text)
                .font(Typography.body)
                .foregroundColor(Color.Design.softWhite)
        }
    }
}

struct ProductButton: View {
    let product: Product
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack {
                VStack(alignment: .leading) {
                    Text(product.displayName)
                        .font(Typography.headline)
                        .foregroundColor(Color.Design.softWhite)
                    Text(product.description)
                        .font(Typography.caption)
                        .foregroundColor(Color.Design.mutedGray)
                }
                Spacer()
                Text(product.displayPrice)
                    .font(Typography.headline)
                    .foregroundColor(Color.Design.sunriseGold)
            }
            .padding()
            .background(
                RoundedRectangle(cornerRadius: CornerRadius.md)
                    .fill(Color.Design.darkIndigo.opacity(0.6))
                    .overlay(
                        RoundedRectangle(cornerRadius: CornerRadius.md)
                            .stroke(Color.Design.sunriseGold.opacity(0.3), lineWidth: 1)
                    )
            )
        }
    }
}
