import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(output):
    # derivative of sigmoid, expressed in terms of its OWN output
    # (this trick avoids recomputing sigmoid from scratch)
    # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    return output * (1 - output)


class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs) * 0.01
        self.bias = 0.0

    def forward(self, inputs):
        # save inputs/output — backward() needs them to compute gradients
        self.inputs = np.array(inputs)
        raw = np.dot(self.weights, self.inputs) + self.bias
        self.output = sigmoid(raw)
        return self.output

    def backward(self, target, learning_rate=0.1):
        """
        One step of backpropagation (just the chain rule) + gradient descent.
        Must be called AFTER forward(), since it reuses self.inputs/self.output.
        """
        # 1. how wrong were we? (simple difference — good enough for one neuron)
        error = self.output - target

        # 2. how much does a nudge in the RAW score move the OUTPUT?
        #    chain rule: d(loss)/d(raw) = d(loss)/d(output) * d(output)/d(raw)
        d_output = error * sigmoid_derivative(self.output)

        # 3. how much did each weight actually contribute to that raw score?
        #    (raw = w . x + b, so d(raw)/d(w_i) = x_i)
        d_weights = d_output * self.inputs
        d_bias = d_output

        # 4. nudge weights/bias in the direction that REDUCES the error
        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias

        return abs(error)  # returned so we can watch it shrink over training


if __name__ == "__main__":
    # --- Stage 1 sanity check (unchanged) ---
    print(sigmoid(0))

    n = Neuron(3)
    output = n.forward([1.0, 0.5, -1.0])
    print(f"Neuron output before training: {output:.4f}")

    # --- Stage 2: teach this single neuron one pattern ---
    # goal: given this exact input, learn to output close to 1.0
    training_input = [1.0, 0.5, -1.0]
    target = 1.0

    print("\nTraining on one example...")
    for epoch in range(1000):
        output = n.forward(training_input)
        error = n.backward(target, learning_rate=0.1)
        if epoch % 100 == 0:
            print(f"epoch {epoch:4d} | output={output:.4f} | error={error:.4f}")

    final_output = n.forward(training_input)
    print(f"\nFinal output: {final_output:.4f} (target was {target})")