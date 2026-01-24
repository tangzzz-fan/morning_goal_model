import CoreML
import Foundation
import OSLog

/// 增强的中文BERT分词器
final class EnhancedChineseBertTokenizer {
    private let vocab: [String: Int32]
    private let clsId: Int32
    private let sepId: Int32
    private let padId: Int32
    private let unkId: Int32
    private let maxTokenLength: Int = 512
    private let logger = Logger(subsystem: "com.morninggoal.app", category: "enhanced-tokenizer")

    // 中文特定处理
    private let chineseStopWords: Set<String>
    private let sentimentWords: [String: Double]
    private let categoryKeywords: [String: [String]]

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

        // 初始化中文处理资源
        self.chineseStopWords = Self.loadChineseStopWords()
        self.sentimentWords = Self.loadSentimentWords()
        self.categoryKeywords = Self.loadCategoryKeywords()

        logger.log("initialized_enhanced_chinese_tokenizer vocab_size=\(self.vocab.count)")
    }

    func encode(_ text: String, maxLength: Int) -> EnhancedTokenizedOutput {
        logger.log("encoding_text text='\(text)' max_length=\(maxLength)")

        // 增强的预处理
        let processedText = enhancedPreprocess(text)
        logger.log("preprocessed_text='\(processedText)'")

        // 智能分词
        let words = intelligentSegmentation(processedText)
        logger.log("segmented_words=\(words.joined(separator: "|"))")

        // 词性标注和关键词提取
        let enhancedWords = enhanceWithPOSAndKeywords(words)

        // WordPiece分词
        let subwordTokens = applyAdvancedWordPiece(enhancedWords)
        logger.log("wordpiece_tokens=\(subwordTokens.joined(separator: "|"))")

        // 构建最终token序列
        var tokenIds: [Int32] = [clsId]
        var tokens = ["[CLS]"]
        var enhancedFeatures: [EnhancedFeature] = []

        for (index, token) in subwordTokens.enumerated() {
            if tokenIds.count >= maxLength - 1 { break }

            if let id = vocab[token] {
                tokenIds.append(id)
                tokens.append(token)

                // 添加增强特征
                let feature = extractEnhancedFeatures(for: token, position: index, originalWords: words)
                enhancedFeatures.append(feature)
            } else {
                tokenIds.append(unkId)
                tokens.append("[UNK]")
                enhancedFeatures.append(EnhancedFeature())
            }
        }

        tokenIds.append(sepId)
        tokens.append("[SEP]")

        // 填充到最大长度
        if tokenIds.count < maxLength {
            let padCount = maxLength - tokenIds.count
            tokenIds += Array(repeating: padId, count: padCount)
            tokens += Array(repeating: "[PAD]", count: padCount)
            enhancedFeatures += Array(repeating: EnhancedFeature(), count: padCount)
        }

        let mask: [Int32] = tokenIds.map { $0 == padId ? 0 : 1 }

        return EnhancedTokenizedOutput(
            ids: tokenIds,
            mask: mask,
            tokens: tokens,
            enhancedFeatures: enhancedFeatures,
            sentimentScore: calculateSentimentScore(words),
            categoryHints: extractCategoryHints(words),
            keywords: extractKeywords(words)
        )
    }

    // MARK: - 增强预处理

    private func enhancedPreprocess(_ text: String) -> String {
        var result = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // 中文特定的文本规范化
        result = normalizeChineseText(result)

        // 移除停用词
        result = removeStopWords(result)

        // 同义词替换
        result = replaceSynonyms(result)

        // 情感词增强
        result = enhanceSentimentWords(result)

        return result
    }

    private func normalizeChineseText(_ text: String) -> String {
        var normalized = text

        // 繁简转换（简化版）
        let traditionalSimplified: [String: String] = [
            "體": "体", "會": "会", "來": "来", "對": "对",
            "說": "说", "時": "时", "個": "个", "為": "为"
        ]

        for (traditional, simplified) in traditionalSimplified {
            normalized = normalized.replacingOccurrences(of: traditional, with: simplified)
        }

        // 数字标准化
        let numberWords = [
            "一": "1",
            "二": "2",
            "三": "3",
            "四": "4",
            "五": "5",
            "六": "6",
            "七": "7",
            "八": "8",
            "九": "9",
            "十": "10"
        ]

        for (word, number) in numberWords {
            normalized = normalized.replacingOccurrences(of: word, with: number)
        }

        return normalized
    }

    private func removeStopWords(_ text: String) -> String {
        var words = text.split(separator: "").map(String.init)
        words = words.filter { !chineseStopWords.contains($0) }
        return words.joined()
    }

    private func replaceSynonyms(_ text: String) -> String {
        let synonyms: [String: [String]] = [
            "工作": ["任务", "项目", "职责"],
            "学习": ["读书", "研究", "进修"],
            "健康": ["身体", "健身", "养生"],
            "家庭": ["家人", "亲属", "亲戚"]
        ]

        var result = text
        for (key, values) in synonyms {
            if let matched = values.first(where: { result.contains($0) }) {
                result = result.replacingOccurrences(of: matched, with: key)
            }
        }
        return result
    }

    private func enhanceSentimentWords(_ text: String) -> String {
        var result = text

        // 强化情感表达
        let sentimentEnhancers: [String: String] = [
            "好": "很好", "坏": "很坏", "开心": "非常开心",
            "难过": "非常难过", "累": "很累", "忙": "很忙"
        ]

        for (original, enhanced) in sentimentEnhancers {
            if result.contains(original) && !result.contains(enhanced) {
                result = result.replacingOccurrences(of: original, with: enhanced)
            }
        }

        return result
    }

    // MARK: - 智能分词

    private func intelligentSegmentation(_ text: String) -> [String] {
        var words: [String] = []
        var currentWord = ""

        // 基于词典的最大匹配分词
        let dictionary = Set(vocab.keys)

        for char in text {
            currentWord.append(char)

            // 检查当前词是否在词典中
            if dictionary.contains(currentWord) {
                continue
            } else if currentWord.count > 1 {
                // 如果当前词不在词典中，尝试前一个词
                let previousWord = String(currentWord.dropLast())
                if dictionary.contains(previousWord) || previousWord.count == 1 {
                    words.append(previousWord)
                    currentWord = String(char)
                }
            }
        }

        if !currentWord.isEmpty {
            words.append(currentWord)
        }

        return words.filter { !$0.isEmpty }
    }

    // MARK: - 词性标注和关键词提取

    private func enhanceWithPOSAndKeywords(_ words: [String]) -> [String] {
        var enhancedWords = words

        // 添加词性标记（简化版）
        for i in 0 ..< words.count {
            let word = words[i]

            // 动词标记
            if word.hasSuffix("了") || word.hasSuffix("着") || word.hasSuffix("过") {
                enhancedWords[i] = word + "_V"
            }
            // 形容词标记
            else if word.hasSuffix("的") && word.count > 1 {
                enhancedWords[i] = word + "_ADJ"
            }
            // 名词标记
            else if word.count <= 3 && i < words.count - 1 {
                enhancedWords[i] = word + "_N"
            }
        }

        return enhancedWords
    }

    // MARK: - 高级WordPiece

    private func applyAdvancedWordPiece(_ words: [String]) -> [String] {
        var result: [String] = []

        for word in words {
            // 移除词性标记进行分词
            let cleanWord = word.replacingOccurrences(of: "_[A-Z]+", with: "", options: .regularExpression)

            if vocab[cleanWord] != nil {
                result.append(cleanWord)
                continue
            }

            // 应用WordPiece算法
            let subwords = applyWordPieceToWord(cleanWord)
            result.append(contentsOf: subwords)
        }

        return result
    }

    private func applyWordPieceToWord(_ word: String) -> [String] {
        var result: [String] = []
        var remaining = word

        while !remaining.isEmpty && remaining.count > 1 {
            var found = false

            // 从长到短尝试所有可能的子词
            for length in stride(from: remaining.count, to: 0, by: -1) {
                let substring = String(remaining.prefix(length))
                let searchToken = result.isEmpty ? substring : "##" + substring

                if vocab[searchToken] != nil {
                    result.append(searchToken)
                    remaining = String(remaining.dropFirst(length))
                    found = true
                    break
                }
            }

            if !found {
                result.append("[UNK]")
                break
            }
        }

        return result.isEmpty ? ["[UNK]"] : result
    }

    // MARK: - 特征提取

    private func extractEnhancedFeatures(for token: String, position: Int, originalWords: [String]) -> EnhancedFeature {
        var feature = EnhancedFeature()

        // 词频特征
        feature.frequency = calculateFrequency(token, in: originalWords)

        // 位置特征
        feature.position = Double(position) / Double(originalWords.count)

        // 长度特征
        feature.length = Double(token.count)

        // 情感特征
        if let sentimentScore = sentimentWords[token] {
            feature.sentimentScore = sentimentScore
        }

        return feature
    }

    private func calculateSentimentScore(_ words: [String]) -> Double {
        var totalScore = 0.0
        var count = 0

        for word in words {
            if let score = sentimentWords[word] {
                totalScore += score
                count += 1
            }
        }

        return count > 0 ? totalScore / Double(count) : 0.0
    }

    private func extractCategoryHints(_ words: [String]) -> [String: Double] {
        var hints: [String: Double] = [:]

        for (category, keywords) in categoryKeywords {
            var score = 0.0
            for word in words where keywords.contains(word) {
                score += 1.0
            }
            if score > 0 {
                hints[category] = score / Double(words.count)
            }
        }

        return hints
    }

    private func extractKeywords(_ words: [String]) -> [String] {
        // 简单的关键词提取：长度大于2且不在停用词中的词
        return words.filter { word in
            word.count > 2 && !chineseStopWords.contains(word)
        }
    }

    private func calculateFrequency(_ token: String, in words: [String]) -> Double {
        let count = words.filter { $0.contains(token) }.count
        return Double(count) / Double(words.count)
    }

    // MARK: - 静态资源加载

    private static func loadChineseStopWords() -> Set<String> {
        return Set([
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
            "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
            "自己", "这", "那", "些", "个", "只", "现在", "时候", "今天", "明天", "昨天"
        ])
    }

    private static func loadSentimentWords() -> [String: Double] {
        return [
            "好": 0.8, "很好": 1.0, "棒": 1.0, "优秀": 1.0, "完美": 1.0,
            "坏": -0.8, "很差": -1.0, "糟糕": -1.0, "失败": -1.0,
            "开心": 1.0, "快乐": 1.0, "高兴": 1.0, "兴奋": 1.0,
            "难过": -1.0, "伤心": -1.0, "失望": -0.8, "沮丧": -0.8,
            "累": -0.5, "疲惫": -0.6, "辛苦": -0.4,
            "轻松": 0.6, "舒服": 0.7, "愉快": 0.8
        ]
    }

    private static func loadCategoryKeywords() -> [String: [String]] {
        return [
            "工作": ["工作", "任务", "项目", "职责", "职业", "事业", "上班", "办公", "业务", "工作场所"],
            "健康": ["健康", "身体", "健身", "运动", "锻炼", "营养", "饮食", "睡眠", "休息", "医疗"],
            "家庭": ["家庭", "家人", "父母", "子女", "孩子", "配偶", "亲戚", "家务", "房子", "家"],
            "学习": ["学习", "读书", "研究", "进修", "培训", "教育", "知识", "技能", "课程", "学校"],
            "财务": ["财务", "金钱", "收入", "支出", "预算", "投资", "理财", "银行", "消费", "购物"],
            "社交": ["社交", "朋友", "同事", "聚会", "活动", "交流", "沟通", "关系", "人脉", "社交活动"],
            "休闲": ["休闲", "娱乐", "放松", "游戏", "电影", "音乐", "旅游", "度假", "爱好", "兴趣"],
            "个人发展": ["个人发展", "成长", "进步", "提升", "完善", "实现", "目标", "梦想", "追求", "自我提升"]
        ]
    }
}

// MARK: - 增强的输出结构

struct EnhancedTokenizedOutput {
    let ids: [Int32]
    let mask: [Int32]
    let tokens: [String]
    let enhancedFeatures: [EnhancedFeature]
    let sentimentScore: Double
    let categoryHints: [String: Double]
    let keywords: [String]
}

struct EnhancedFeature {
    var frequency: Double = 0.0
    var position: Double = 0.0
    var length: Double = 0.0
    var sentimentScore: Double = 0.0
    var isKeyword: Bool = false
    var categoryHint: String?
}
