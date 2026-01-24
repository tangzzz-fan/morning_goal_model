import Foundation
import StoreKit

@MainActor
@Observable
class StoreManager {
    static let shared = StoreManager()

    var products: [Product] = []
    var purchasedProductIDs: Set<String> = []
    var isLoading = false
    var errorMessage: String?

    // Product IDs
    private let productDict: [String: String] = [
        "com.morninggoal.lifetime": "Lifetime Access",
        "com.morninggoal.yearly": "Yearly Subscription"
    ]

    init() {
        // Check for existing purchases on launch
        Task {
            await updatePurchasedProducts()
        }
    }

    func loadProducts() async {
        isLoading = true
        errorMessage = nil

        do {
            let products = try await Product.products(for: productDict.keys)
            self.products = products
                .sorted { $0.price < $1.price } // Sort by price usually puts subscription first or last depending on logic, let's just sort.
            isLoading = false
        } catch {
            print("Failed to load products: \(error)")
            errorMessage = "Failed to load products. Please try again."
            isLoading = false
        }
    }

    func purchase(_ product: Product) async throws {
        let result = try await product.purchase()

        switch result {
        case let .success(verification):
            // Check if the transaction is verified
            switch verification {
            case let .verified(transaction):
                // Update local state
                await updatePurchasedProducts()

                // Always finish a transaction.
                await transaction.finish()

            case let .unverified(_, error):
                // Successful purchase but transaction/receipt can't be verified
                // Could be a jailbroken phone
                print("Unverified transaction: \(error)")
                throw StoreError.failedVerification
            }

        case .userCancelled:
            break

        case .pending:
            // Transaction waiting on SCA (Strong Customer Authentication) or
            // approval from Ask to Buy
            break

        @unknown default:
            break
        }
    }

    func restorePurchases() async {
        // In StoreKit 2, AppStore.sync() is rarely needed.
        // Just re-checking current entitlements is usually enough.
        // However, for UI feedback, we can call sync() if needed, but usually just updating state is fine.
        try? await AppStore.sync()
        await updatePurchasedProducts()
    }

    private func updatePurchasedProducts() async {
        var purchased = Set<String>()

        // Iterate through the user's current entitlements
        for await result in Transaction.currentEntitlements {
            switch result {
            case let .verified(transaction):
                // Check if the transaction is revoked
                if transaction.revocationDate == nil {
                    purchased.insert(transaction.productID)
                }
            case .unverified:
                break
            }
        }

        self.purchasedProductIDs = purchased
    }

    var isPro: Bool {
        return !purchasedProductIDs.isEmpty
    }
}

enum StoreError: Error {
    case failedVerification
}
