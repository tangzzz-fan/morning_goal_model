import Foundation

enum FeatureEngineeringUtils {
    static func segmentWords(_ text: String) -> [String] {
        var words: [String] = []
        var currentWord = ""

        for char in text {
            if char.isLetter || char.isNumber {
                currentWord.append(char)
            } else {
                if !currentWord.isEmpty {
                    words.append(currentWord)
                    currentWord = ""
                }
            }
        }

        if !currentWord.isEmpty {
            words.append(currentWord)
        }

        return words.filter { !$0.isEmpty }
    }

    static func extractKeywords(_ words: [String]) -> [String] {
        words.filter { word in
            word.count > 2 && !FeatureLexicon.stopWords.contains(word)
        }
    }
}
