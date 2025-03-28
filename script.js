// Example trained parameters (manually copy from Python JSON output)
const weights = {
    hidden1: { w: [0.1, 0.2, -0.1, 0.5, -0.2, 0.3], b: 0.1 },
    hidden2: { w: [0.4, -0.5, 0.2, 0.1, -0.3, 0.6], b: -0.1 },
    output: { w: [0.2, -0.3, 0.5, 0.1, -0.4, 0.7], b: 0.2 }
};

// Activation function (ReLU)
function relu(x) {
    return Math.max(0, x);
}

// Function to predict success
function predictMovieSuccess(inputs) {
    // Convert inputs to numbers
    let X = inputs.map(Number);

    // Forward propagation
    let hidden1_output = relu(X.reduce((sum, x, i) => sum + x * weights.hidden1.w[i], weights.hidden1.b));
    let hidden2_output = relu(X.reduce((sum, x, i) => sum + x * weights.hidden2.w[i], weights.hidden2.b));
    
    let final_output = hidden1_output * weights.output.w[0] + hidden2_output * weights.output.w[1] + weights.output.b;
    
    // Convert to percentage (Success Rate)
    let successRate = Math.min(100, Math.max(0, final_output * 100)); // Scale to 0-100%
    
    return successRate.toFixed(2) + "%";
}

// Handle form submission
document.getElementById("movieForm").addEventListener("submit", function(event) {
    event.preventDefault();

    let inputs = [
        document.getElementById("budget").value,
        document.getElementById("screen_count").value,
        document.getElementById("director_success").value,
        document.getElementById("actor_popularity").value,
        document.getElementById("past_success").value,
        document.getElementById("music_popularity").value
    ];

    let prediction = predictMovieSuccess(inputs);
    document.getElementById("predictionResult").innerText = prediction;
});
