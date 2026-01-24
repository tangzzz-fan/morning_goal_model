import CoreML
import Foundation
import OSLog

/// 智能特征工程服务
final class IntelligentFeatureEngineeringService {
    private let logger = Logger(subsystem: "com.morninggoal.app", category: "feature-engineering")

    struct TextFeatures {
        let basicFeatures: BasicFeatures
        let semanticFeatures: SemanticFeatures
        let sentimentFeatures: SentimentFeatures
        let categoryFeatures: CategoryFeatures
        let statisticalFeatures: StatisticalFeatures
    }

    struct BasicFeatures {
        let length: Int
        let wordCount: Int
        let avgWordLength: Double
        let punctuationCount: Int
        let digitCount: Int
        let chineseCharRatio: Double
    }

    struct SemanticFeatures {
        let keyWords: [String]
        let keyPhrases: [String]
        let semanticSimilarity: [String: Double]
        let wordEmbeddings: [Double]
    }

    struct SentimentFeatures {
        let positiveWords: Int
        let negativeWords: Int
        let neutralWords: Int
        let sentimentScore: Double
        let emotionCategories: [String: Double]
    }

    struct CategoryFeatures {
        let workKeywords: Int
        let healthKeywords: Int
        let familyKeywords: Int
        let studyKeywords: Int
        let financeKeywords: Int
        let socialKeywords: Int
        let leisureKeywords: Int
        let personalKeywords: Int
        let categoryScores: [String: Double]
    }

    struct StatisticalFeatures {
        let tfidfScores: [String: Double]
        let wordFrequency: [String: Int]
        let ngramFeatures: [String: Int]
        let posTagDistribution: [String: Double]
    }

    func extractFeatures(from text: String) -> TextFeatures {
        logger.log("extracting_features_from_text text='\(text)'")

        let startTime = CFAbsoluteTimeGetCurrent()

        let basicFeatures = extractBasicFeatures(text)
        let semanticFeatures = extractSemanticFeatures(text)
        let sentimentFeatures = extractSentimentFeatures(text)
        let categoryFeatures = extractCategoryFeatures(text)
        let statisticalFeatures = extractStatisticalFeatures(text)

        let endTime = CFAbsoluteTimeGetCurrent()
        logger.log("feature_extraction_completed time=\(String(format: "%.3f", endTime - startTime))s")

        return TextFeatures(
            basicFeatures: basicFeatures,
            semanticFeatures: semanticFeatures,
            sentimentFeatures: sentimentFeatures,
            categoryFeatures: categoryFeatures,
            statisticalFeatures: statisticalFeatures
        )
    }

    private func extractBasicFeatures(_ text: String) -> BasicFeatures {
        let length = text.count
        let words = FeatureEngineeringUtils.segmentWords(text)
        let wordCount = words.count
        let avgWordLength = wordCount > 0 ? Double(words.map { $0.count }.reduce(0, +)) / Double(wordCount) : 0.0
        let punctuationCount = text.filter { "，。！？；：".contains($0) }.count
        let digitCount = text.filter { $0.isNumber }.count
        let chineseCharRatio = Double(text.filter { $0.unicodeScalars.first?.value ?? 0 > 0x4E00 && $0.unicodeScalars.first?.value ?? 0 < 0x9FFF }
            .count) / Double(length)

        return BasicFeatures(
            length: length,
            wordCount: wordCount,
            avgWordLength: avgWordLength,
            punctuationCount: punctuationCount,
            digitCount: digitCount,
            chineseCharRatio: chineseCharRatio
        )
    }

    private func extractSemanticFeatures(_ text: String) -> SemanticFeatures {
        let words = FeatureEngineeringUtils.segmentWords(text)
        let keyWords = FeatureEngineeringUtils.extractKeywords(words)
        let keyPhrases = extractKeyPhrases(words)
        let semanticSimilarity = calculateSemanticSimilarity(words)
        let wordEmbeddings = generateSimpleWordEmbeddings(words)

        return SemanticFeatures(
            keyWords: keyWords,
            keyPhrases: keyPhrases,
            semanticSimilarity: semanticSimilarity,
            wordEmbeddings: wordEmbeddings
        )
    }

    private func extractSentimentFeatures(_ text: String) -> SentimentFeatures {
        let words = FeatureEngineeringUtils.segmentWords(text)

        var positiveCount = 0
        var negativeCount = 0
        var neutralCount = 0
        var totalSentimentScore = 0.0

        for word in words {
            if FeatureLexicon.positiveWords.contains(word) {
                positiveCount += 1
                totalSentimentScore += 1.0
            } else if FeatureLexicon.negativeWords.contains(word) {
                negativeCount += 1
                totalSentimentScore -= 1.0
            } else {
                neutralCount += 1
            }
        }

        let sentimentScore = words.isEmpty ? 0.0 : totalSentimentScore / Double(words.count)
        let emotionCategories = calculateEmotionCategories(words)

        return SentimentFeatures(
            positiveWords: positiveCount,
            negativeWords: negativeCount,
            neutralWords: neutralCount,
            sentimentScore: sentimentScore,
            emotionCategories: emotionCategories
        )
    }

    private func extractCategoryFeatures(_ text: String) -> CategoryFeatures {
        let words = FeatureEngineeringUtils.segmentWords(text)

        let workCount = words.filter { FeatureLexicon.workKeywords.contains($0) }.count
        let healthCount = words.filter { FeatureLexicon.healthKeywords.contains($0) }.count
        let familyCount = words.filter { FeatureLexicon.familyKeywords.contains($0) }.count
        let studyCount = words.filter { FeatureLexicon.studyKeywords.contains($0) }.count
        let financeCount = words.filter { FeatureLexicon.financeKeywords.contains($0) }.count
        let socialCount = words.filter { FeatureLexicon.socialKeywords.contains($0) }.count
        let leisureCount = words.filter { FeatureLexicon.leisureKeywords.contains($0) }.count
        let personalCount = words.filter { FeatureLexicon.personalKeywords.contains($0) }.count

        let counts = [
            "工作": workCount,
            "健康": healthCount,
            "家庭": familyCount,
            "学习": studyCount,
            "财务": financeCount,
            "社交": socialCount,
            "休闲": leisureCount,
            "个人发展": personalCount
        ]
        let categoryScores = calculateCategoryScores(counts, totalWords: words.count)

        return CategoryFeatures(
            workKeywords: workCount,
            healthKeywords: healthCount,
            familyKeywords: familyCount,
            studyKeywords: studyCount,
            financeKeywords: financeCount,
            socialKeywords: socialCount,
            leisureKeywords: leisureCount,
            personalKeywords: personalCount,
            categoryScores: categoryScores
        )
    }

    private func extractStatisticalFeatures(_ text: String) -> StatisticalFeatures {
        let words = FeatureEngineeringUtils.segmentWords(text)
        let tfidfScores = calculateTFIDF(words)
        let wordFrequency = calculateWordFrequency(words)
        let ngramFeatures = extractNgramFeatures(words)
        let posTagDistribution = calculatePOSTagDistribution(text)

        return StatisticalFeatures(
            tfidfScores: tfidfScores,
            wordFrequency: wordFrequency,
            ngramFeatures: ngramFeatures,
            posTagDistribution: posTagDistribution
        )
    }

    // MARK: - 辅助方法

    private func extractKeyPhrases(_ words: [String]) -> [String] {
        var phrases: [String] = []

        // 提取2-gram短语
        for i in 0 ..< words.count - 1 {
            let phrase = words[i] + words[i + 1]
            if phrase.count > 3 {
                phrases.append(phrase)
            }
        }

        return phrases
    }

    private func calculateSemanticSimilarity(_ words: [String]) -> [String: Double] {
        var similarities: [String: Double] = [:]

        // 简化的语义相似度计算
        let keywordCategories = [
            FeatureLexicon.workKeywords,
            FeatureLexicon.healthKeywords,
            FeatureLexicon.familyKeywords,
            FeatureLexicon.studyKeywords,
            FeatureLexicon.financeKeywords,
            FeatureLexicon.socialKeywords,
            FeatureLexicon.leisureKeywords,
            FeatureLexicon.personalKeywords
        ]

        for category in keywordCategories {
            var similarity = 0.0
            for word in words where category.contains(word) {
                similarity += 1.0
            }
            similarities["category_\(keywordCategories.firstIndex(of: category) ?? 0)"] = similarity / Double(words.count)
        }

        return similarities
    }

    private func generateSimpleWordEmbeddings(_ words: [String]) -> [Double] {
        // 简化的词嵌入生成
        var embeddings: [Double] = []

        for word in words {
            // 基于词长和字符的简单嵌入
            let embedding = Double(word.count) * 0.1 + Double(word.unicodeScalars.first?.value ?? 0) * 0.0001
            embeddings.append(embedding)
        }

        // 归一化
        let maxVal = embeddings.max() ?? 1.0
        return embeddings.map { $0 / maxVal }
    }

    private func calculateEmotionCategories(_ words: [String]) -> [String: Double] {
        var emotionScores: [String: Double] = [:]

        for (emotion, keywords) in FeatureLexicon.emotionCategories {
            var score = 0.0
            for word in words where keywords.contains(word) {
                score += 1.0
            }
            emotionScores[emotion] = words.isEmpty ? 0.0 : score / Double(words.count)
        }

        return emotionScores
    }

    private func calculateCategoryScores(_ counts: [String: Int], totalWords: Int) -> [String: Double] {
        var scores: [String: Double] = [:]
        for (category, count) in counts {
            scores[category] = totalWords > 0 ? Double(count) / Double(totalWords) : 0.0
        }
        return scores
    }

    private func calculateTFIDF(_ words: [String]) -> [String: Double] {
        var tfidfScores: [String: Double] = [:]
        let wordFreq = calculateWordFrequency(words)

        for (word, freq) in wordFreq {
            let tf = Double(freq) / Double(words.count)
            let idf = calculateIDF(word)
            tfidfScores[word] = tf * idf
        }

        return tfidfScores
    }

    private func calculateWordFrequency(_ words: [String]) -> [String: Int] {
        var frequency: [String: Int] = [:]

        for word in words {
            frequency[word, default: 0] += 1
        }

        return frequency
    }

    private func calculateIDF(_ word: String) -> Double {
        // 简化的IDF计算
        let totalDocuments = 1000.0 // 假设的文档总数
        let documentsContainingWord = 10.0 // 假设包含该词的文档数
        return log(totalDocuments / documentsContainingWord)
    }

    private func extractNgramFeatures(_ words: [String]) -> [String: Int] {
        var ngrams: [String: Int] = [:]

        // 2-grams
        for i in 0 ..< words.count - 1 {
            let bigram = words[i] + "_" + words[i + 1]
            ngrams[bigram, default: 0] += 1
        }

        // 3-grams
        for i in 0 ..< words.count - 2 {
            let trigram = words[i] + "_" + words[i + 1] + "_" + words[i + 2]
            ngrams[trigram, default: 0] += 1
        }

        return ngrams
    }

    private func calculatePOSTagDistribution(_ text: String) -> [String: Double] {
        // 简化的词性标注分布
        var distribution: [String: Double] = [:]

        let words = FeatureEngineeringUtils.segmentWords(text)
        let totalWords = words.count

        if totalWords == 0 { return distribution }

        // 基于词尾的词性猜测
        var nounCount = 0
        var verbCount = 0
        var adjCount = 0
        var advCount = 0

        for word in words {
            if word.hasSuffix("的") {
                adjCount += 1
            } else if word.hasSuffix("地") {
                advCount += 1
            } else if word.hasSuffix("了") || word.hasSuffix("着") || word.hasSuffix("过") {
                verbCount += 1
            } else {
                nounCount += 1
            }
        }

        distribution["名词"] = Double(nounCount) / Double(totalWords)
        distribution["动词"] = Double(verbCount) / Double(totalWords)
        distribution["形容词"] = Double(adjCount) / Double(totalWords)
        distribution["副词"] = Double(advCount) / Double(totalWords)

        return distribution
    }
}
