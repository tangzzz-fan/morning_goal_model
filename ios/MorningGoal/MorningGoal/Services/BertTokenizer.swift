import Foundation

struct TokenizedOutput {
    let ids: [Int32]
    let mask: [Int32]
    let tokens: [String]
}

final class BertTokenizer {
    private let vocab: [String: Int32]
    private let clsId: Int32
    private let sepId: Int32
    private let padId: Int32
    private let unkId: Int32
    private let maxTokenLength: Int = 512

    init(vocabURL: URL) throws {
        let data = try String(contentsOf: vocabURL, encoding: .utf8)
        var map: [String: Int32] = [:]
        var idx: Int32 = 0
        for line in data.split(separator: "\n", omittingEmptySubsequences: true) {
            map[String(line)] = idx
            idx += 1
        }
        self.vocab = map
        self.clsId = map["[CLS]"] ?? 101
        self.sepId = map["[SEP]"] ?? 102
        self.padId = map["[PAD]"] ?? 0
        self.unkId = map["[UNK]"] ?? 100
    }

    func encode(_ text: String, maxLength: Int) -> TokenizedOutput {
        let processedText = preprocessText(text)
        let wordTokens = tokenizeWords(processedText)
        let subwordTokens = applyWordPiece(wordTokens)

        var tokenIds: [Int32] = [clsId]
        var tokens = ["[CLS]"]

        for token in subwordTokens {
            if tokenIds.count >= maxLength - 1 { break }
            if let id = vocab[token] {
                tokenIds.append(id)
                tokens.append(token)
            } else {
                tokenIds.append(unkId)
                tokens.append("[UNK]")
            }
        }

        tokenIds.append(sepId)
        tokens.append("[SEP]")

        if tokenIds.count < maxLength {
            let padCount = maxLength - tokenIds.count
            tokenIds += Array(repeating: padId, count: padCount)
            tokens += Array(repeating: "[PAD]", count: padCount)
        }

        let mask: [Int32] = tokenIds.map { $0 == padId ? 0 : 1 }
        return TokenizedOutput(ids: tokenIds, mask: mask, tokens: tokens)
    }

    private func preprocessText(_ text: String) -> String {
        return text.lowercased()
            .replacingOccurrences(of: " ", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func tokenizeWords(_ text: String) -> [String] {
        var words: [String] = []
        var currentWord = ""

        for char in text {
            if char.isPunctuation || char.isSymbol {
                if !currentWord.isEmpty {
                    words.append(currentWord)
                    currentWord = ""
                }
                words.append(String(char))
            } else {
                currentWord.append(char)
            }
        }

        if !currentWord.isEmpty {
            words.append(currentWord)
        }

        return words.filter { !$0.isEmpty }
    }

    private func applyWordPiece(_ words: [String]) -> [String] {
        var result: [String] = []

        for word in words {
            if vocab[word] != nil {
                result.append(word)
                continue
            }

            var remaining = word
            var subwords: [String] = []

            while !remaining.isEmpty && remaining.count > 1 {
                var found = false
                for length in stride(from: remaining.count, to: 0, by: -1) {
                    let substring = String(remaining.prefix(length))
                    let searchToken = subwords.isEmpty ? substring : "##" + substring

                    if vocab[searchToken] != nil {
                        subwords.append(searchToken)
                        remaining = String(remaining.dropFirst(length))
                        found = true
                        break
                    }
                }

                if !found {
                    subwords.append("[UNK]")
                    break
                }
            }

            result.append(contentsOf: subwords)
        }

        return result
    }
}
