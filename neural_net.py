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


class Layer:
    """
    Many neurons side by side, computed as ONE matrix operation
    instead of a Python loop over individual Neuron objects.
    weights shape: (num_neurons, num_inputs) — one row of weights per neuron.
    """

    def __init__(self, num_inputs, num_neurons):
        self.weights = np.random.randn(num_neurons, num_inputs) * 0.5
        self.biases = np.zeros(num_neurons)

    def forward(self, inputs):
        self.inputs = np.array(inputs)
        raw = self.weights @ self.inputs + self.biases  # matrix-vector multiply
        self.output = sigmoid(raw)
        return self.output

    def backward(self, d_loss_d_output, learning_rate=0.1):
        """
        d_loss_d_output: gradient of the loss w.r.t. THIS layer's output,
        handed to us by the layer after us (or by the loss, if we're last).

        Returns: gradient of the loss w.r.t. THIS layer's INPUT, so the
        layer before us can continue the chain rule backward.
        """
        d_raw = d_loss_d_output * sigmoid_derivative(self.output)

        # outer product: every neuron's d_raw paired with every input value
        d_weights = np.outer(d_raw, self.inputs)
        d_biases = d_raw

        # gradient to pass to the PREVIOUS layer, before we update our own weights
        d_inputs = self.weights.T @ d_raw

        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        return d_inputs


class Network:
    """A stack of Layers. layer_sizes=[2, 4, 1] means 2 inputs -> hidden layer
    of 4 neurons -> output layer of 1 neuron."""

    def __init__(self, layer_sizes):
        self.layers = [
            Layer(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ]

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, target, learning_rate=0.1):
        output = self.layers[-1].output
        d_loss_d_output = output - target  # starting point for the chain rule

        # walk backward through every layer, passing the gradient along
        for layer in reversed(self.layers):
            d_loss_d_output = layer.backward(d_loss_d_output, learning_rate)

        return np.abs(output - target).mean()


if __name__ == "__main__":
    # --- Stage 1 sanity check (unchanged) ---
    print(sigmoid(0))

    n = Neuron(3)
    output = n.forward([1.0, 0.5, -1.0])
    print(f"Neuron output before training: {output:.4f}")

    # --- Stage 2: teach this single neuron one pattern ---
    training_input = [1.0, 0.5, -1.0]
    target = 1.0

    print("\nTraining single neuron on one example...")
    for epoch in range(1000):
        output = n.forward(training_input)
        error = n.backward(target, learning_rate=0.1)
        if epoch % 100 == 0:
            print(f"epoch {epoch:4d} | output={output:.4f} | error={error:.4f}")

    final_output = n.forward(training_input)
    print(f"Final output: {final_output:.4f} (target was {target})")

    # --- Stage 3: XOR — the problem a single neuron literally CANNOT learn ---
    # XOR: output is 1 only if the two inputs DIFFER. Not separable by one line,
    # which is exactly why this needs a hidden layer.
    print("\n\nTraining a 2-4-1 network on XOR...")
    xor_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    xor_targets = [0, 1, 1, 0]

    net = Network([2, 4, 1])  # 2 inputs -> hidden layer of 4 -> 1 output

    for epoch in range(5000):
        total_loss = 0
        for x, y in zip(xor_inputs, xor_targets):
            net.forward(x)
            total_loss += net.backward(y, learning_rate=0.5)
        if epoch % 500 == 0:
            print(f"epoch {epoch:4d} | avg loss={total_loss / 4:.4f}")

    print("\nFinal predictions:")
    for x, y in zip(xor_inputs, xor_targets):
        pred = net.forward(x)[0]
        print(f"  input={x} -> predicted={pred:.4f} (target={y})")