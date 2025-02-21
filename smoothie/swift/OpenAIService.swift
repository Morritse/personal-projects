import Foundation

class OpenAIService {
    static let shared = OpenAIService()
    private let apiKey = "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"
    private let endpoint = "https://api.openai.com/v1/chat/completions"
    
    private init() {}
    
    func getSupplementRecommendations(
        age: String,
        weight: String,
        goal: String,
        activityLevel: String,
        completion: @escaping (Result<String, Error>) -> Void
    ) {
        let prompt = """
        Based on the following information, provide specific supplement recommendations with dosages:
        Age: \(age)
        Weight: \(weight) lbs
        Fitness Goal: \(goal)
        Activity Level: \(activityLevel)
        
        Please provide recommendations in the following format:
        1. Supplement name
           - Recommended dosage
           - Timing
           - Purpose
        """
        
        let messages: [[String: Any]] = [
            ["role": "system", "content": """
            You are a knowledgeable fitness and nutrition advisor specializing in supplement recommendations. 
            Focus on evidence-based supplements with proven benefits. 
            Always include important safety notes and potential interactions.
            Consider age, weight, goals, and activity level in your recommendations.
            """],
            ["role": "user", "content": prompt]
        ]
        
        let requestBody: [String: Any] = [
            "model": "gpt-4o",
            "messages": messages,
            "temperature": 0.7
        ]
        
        var request = URLRequest(url: URL(string: endpoint)!)
        request.httpMethod = "POST"
        request.addValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: requestBody)
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    completion(.failure(error))
                    return
                }
                
                guard let data = data else {
                    completion(.failure(NSError(domain: "", code: -1, userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                    return
                }
                
                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let choices = json["choices"] as? [[String: Any]],
                       let firstChoice = choices.first,
                       let message = firstChoice["message"] as? [String: Any],
                       let content = message["content"] as? String {
                        completion(.success(content))
                    } else {
                        throw NSError(domain: "", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid response format"])
                    }
                } catch {
                    completion(.failure(error))
                }
            }
        }.resume()
    }
}
