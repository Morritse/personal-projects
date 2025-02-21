import SwiftUI

struct UserData {
    var age: String = ""
    var weight: String = ""
    var goal: String = "Muscle Gain"
    var activityLevel: String = "Moderate"
}

struct ContentView: View {
    @State private var userData = UserData()
    @State private var recommendations: String = ""
    @State private var isLoading = false
    
    let goals = ["Muscle Gain", "Fat Loss", "Athletic Performance", "General Health"]
    let activityLevels = ["Sedentary", "Light", "Moderate", "Very Active", "Extremely Active"]
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Personal Information")) {
                    TextField("Age", text: $userData.age)
                        .keyboardType(.numberPad)
                    TextField("Weight (lbs)", text: $userData.weight)
                        .keyboardType(.numberPad)
                    Picker("Fitness Goal", selection: $userData.goal) {
                        ForEach(goals, id: \.self) {
                            Text($0)
                        }
                    }
                    Picker("Activity Level", selection: $userData.activityLevel) {
                        ForEach(activityLevels, id: \.self) {
                            Text($0)
                        }
                    }
                }
                
                Section {
                    Button(action: getRecommendations) {
                        if isLoading {
                            ProgressView()
                        } else {
                            Text("Get Recommendations")
                        }
                    }
                }
                
                if !recommendations.isEmpty {
                    Section(header: Text("Supplement Recommendations")) {
                        Text(recommendations)
                    }
                }
            }
            .navigationTitle("Supplement Advisor")
        }
    }
    
    func getRecommendations() {
        guard !userData.age.isEmpty, !userData.weight.isEmpty else {
            recommendations = "Please fill in all fields"
            return
        }
        
        isLoading = true
        
        // Here we'll integrate with OpenAI API
        OpenAIService.shared.getSupplementRecommendations(
            age: userData.age,
            weight: userData.weight,
            goal: userData.goal,
            activityLevel: userData.activityLevel
        ) { result in
            isLoading = false
            switch result {
            case .success(let response):
                recommendations = response
            case .failure(let error):
                recommendations = "Error: \(error.localizedDescription)"
            }
        }
    }
}

#Preview {
    ContentView()
}
