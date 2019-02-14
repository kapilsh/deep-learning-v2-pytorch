from my_answers import NeuralNetwork
import numpy as np


if __name__ == '__main__':

    X = np.arange(3).reshape(1, 3)
    y = np.array([[2]])
    hidden = 4
    learn_rate = 0.01

    print(f"X: {X} ; y: {y}")

    nn = NeuralNetwork(input_nodes=X.shape[1], hidden_nodes=hidden,
                       output_nodes=1, learning_rate=learn_rate)

    final, hidden = nn.forward_pass_train(X)

    # print(f"Hidden Output {hidden.shape}: {hidden}")
    #
    # print(f"Final Output: {final.shape}: {final}")

    delta_weights_i_h = np.zeros(nn.weights_input_to_hidden.shape)
    delta_weights_h_o = np.zeros(nn.weights_hidden_to_output.shape)

    delta_weights_i_h, delta_weights_h_o = nn.backpropagation(
        X=X, y=y, final_outputs=final, hidden_outputs=hidden,
        delta_weights_i_h=delta_weights_i_h,
        delta_weights_h_o=delta_weights_h_o)

    # print(delta_weights_i_h, delta_weights_h_o)




