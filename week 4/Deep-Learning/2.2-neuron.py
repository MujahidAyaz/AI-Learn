import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        """Initialize the neuron with specific weights and bias."""
        self.weights = np.array(weights, dtype=np.float32)
        self.bias = float(bias)
        
    def relu(self, x):
        """ReLU Activation Function"""
        return np.maximum(0, x)
        
    def forward(self, inputs):
        """Perform forward pass and print all intermediate values (Task 1)."""
        inputs = np.array(inputs, dtype=np.float32)
        weighted_sum = np.dot(inputs, self.weights)
        z = weighted_sum + self.bias
        output = self.relu(z)
        
        # Printing intermediate values for Task 1
        print("----- Processing Student -----")
        print(f"Inputs:       {inputs}")
        print(f"Weights:      {self.weights}")
        print(f"Weighted Sum: {weighted_sum:.2f}")
        print(f"Bias:         {self.bias:.2f}")
        print(f"Output:       {output:.2f}\n")
        
        return output

# --- Task 2: Run the neuron using three different students ---

# Shared network parameters
weights_data = [0.4, 0.7, 0.2]
bias_data = 1.0

# Student data
students = {
    "Student 1": [5, 8, 6],
    "Student 2": [7, 10, 9],
    "Student 3": [2, 4, 3]
}

# Instantiate the OOP neuron
neural_neuron = Neuron(weights=weights_data, bias=bias_data)

# Process each student
for student_name, student_inputs in students.items():
    print(f"Evaluating {student_name}:")
    prediction = neural_neuron.forward(student_inputs)
