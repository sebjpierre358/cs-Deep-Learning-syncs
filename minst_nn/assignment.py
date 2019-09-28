from __future__ import absolute_import
from matplotlib import pyplot as plt
import numpy as np
from preprocess import get_data

class Model:
    """
    This model class will contain the architecture for
    your single layer Neural Network for classifying MNIST with
    batched learning. Please implement the TODOs for the entire
    model but do not change the method and constructor arguments.
    Make sure that your Model class works with multiple batch
    sizes. Additionally, please exclusively use NumPy and
    Python built-in functions for your implementation.
    """

    def __init__(self):
        # TODO: Initialize all hyperparametrs
        self.input_size = 784 # Size of image vectors
        self.num_classes = 10 # Number of classes/possible labels
        self.batch_size = 100
        self.learning_rate = .5

        # TODO: Initialize weights and biases
        self.W = np.zeros((self.num_classes, self.input_size))
        self.b = np.zeros(self.num_classes)

    def call(self, inputs):
        """
        Does the forward pass on an batch of input images.
        :param inputs: normalized (0.0 to 1.0) batch of images,
                       (batch_size x 784) (2D), where batch can be any number.
        :return: probabilities, probabilities for each class per image # (batch_size x 10)
        """
        # TODO: Write the forward pass logic for your model
        # TODO: Calculate, then return, the probability for each class per image using the Softmax equation
        #self.batch_size = inputs.shape[0]
        self.batch_size = inputs.shape[0]

        L = np.ndarray((self.batch_size, self.num_classes), dtype=np.float32)
        for i in range(self.batch_size):
            l = np.sum((self.W * inputs[i]), axis=1)
            l = np.add(l, self.b)
            L[i] = l.transpose()

        L = np.exp(L.transpose())
        P = np.transpose(L / np.sum(L, axis=0))

        return P

    def loss(self, probabilities, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Loss should be decreasing with every training loop (step).
        NOTE: This function is not actually used for gradient descent
        in this assignment, but is a sanity check to make sure model
        is learning.
        :param probabilities: matrix that contains the probabilities
        per class per image of each class for each image
        :param labels: the true batch labels
        :return: average loss per batch element (float)
        """
        # TODO: calculate average cross entropy loss for a batch
        total_loss = 0
        for e in range(self.batch_size):
            #print("label of image " + str(e) + " is " + str(labels[e]))

            #print("probability guessed " + str(labels[e]) + ": " +
            #str(probabilities[e][labels[e]]))

            error = -np.log(probabilities[e][labels[e]])
            #print("loss for example " + str(e) + ": " + str(error) + "\n")

            total_loss += error

        #print("TOTAL loss this batch: " + str(total_loss))

        avg_loss = total_loss / self.batch_size
        #print("AVERAGE LOSS THIS BATCH: " + str(avg_loss))

        return avg_loss

    def back_propagation(self, inputs, probabilities, labels):
        """
        Returns the gradients for model's weights and biases
        after one forward pass and loss calculation. The learning
        algorithm for updating weights and biases mentioned in
        class works for one image, but because we are looking at
        batch_size number of images at each step, you should take the
        average of the gradients across all images in the batch.
        :param inputs: batch inputs (a batch of images)
        :param probabilities: matrix that contains the probabilities of each
        class for each image
        :param labels: true labels
        :return: gradient for weights,and gradient for biases
        """
        # TODO: calculate the gradients for the weights and the gradients for the bias with respect to average loss
        #print(probabilities.shape)
        one_hot = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)
        img_probs = np.ndarray((self.batch_size,))
        for i in range(self.batch_size):
            label = labels[i]
            one_hot[i][label] = 1.0
            #img_probs[i] = probabilities[i][label]

        dp_dl = np.transpose(np.transpose((one_hot - probabilities))) #*img_probs)
        #pseudocode doesn't mulitiply by img_probs, so hmmmmm

        gradB = np.sum(dp_dl, 0) * self.learning_rate / self.batch_size
        gradW = np.zeros((self.input_size, self.num_classes), dtype=np.float32)

        for i in range(self.batch_size):
            for j in range(self.input_size):
                gradW[j] = gradW[j] + (dp_dl[i] * inputs[i][j] * self.learning_rate)

        gradW = np.transpose(gradW * (1/self.batch_size)) #average the gradient
        return (gradW, gradB)

    def accuracy(self, probabilities, labels):
        """
        Calculates the model's accuracy by comparing the number
        of correct predictions with the correct answers.
        :param probabilities: result of running model.call() on test inputs
        :param labels: test set labels
        :return: Float (0,1) that contains batch accuracy
        """
        # TODO: calculate the batch accuracy
        num_correct = 0.0
        model_predictions = np.argmax(probabilities, axis=1)

        for i in range(self.batch_size):
            if model_predictions[i] == labels[i]:
                num_correct += 1.0

        return num_correct / self.batch_size

    def gradient_descent(self, gradW, gradB):
        '''
        Given the gradients for weights and biases, does gradient
        descent on the Model's parameters.
        :param gradW: gradient for weights
        :param gradB: gradient for biases
        :return: None
        '''
        self.W = self.W + gradW
        self.b = self.b + gradB

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels.
    :param model: the initialized model to use for the forward
    pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training)
    :param train_inputs: train labels (all labels to use for training)
    :return: None
    '''

    # TODO: Iterate over the training inputs and labels, in model.batch_size increments
    # TODO: For every batch, compute then descend the gradients for the model's weights
    batch = 0
    #model.batch_size = train_inputs.shape[0]

    while batch < train_labels.size:
        batch_inputs = train_inputs[batch : batch + model.batch_size]
        batch_labels = train_labels[batch : batch + model.batch_size]

        probs = model.call(batch_inputs)
        #model.loss(probs, batch_labels)
        gradW, gradB = model.back_propagation(batch_inputs, probs, batch_labels)
        model.gradient_descent(gradW, gradB)

        batch += model.batch_size

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. For this assignment,
    the inputs should be the entire test set, but in the future we will
    ask you to batch it instead.
    :param test_inputs: MNIST test data (all images to be tested)
    :param test_labels: MNIST test labels (all corresponding labels)
    :return: accuracy - Float (0,1)
    """
    # TODO: Iterate over the testing inputs and labels
    # TODO: Return accuracy across testing set
    probs = model.call(test_inputs)

    return model.accuracy(probs, test_labels)


def visualize_results(image_inputs, probabilities, image_labels):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data()
    :param probabilities: the output of model.call()
    :param image_labels: the labels from get_data()

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    images = np.reshape(image_inputs, (-1, 28, 28))
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = images.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(images[ind], cmap="Greys")
        ax.set(title="PL: {}\nAL: {}".format(predicted_labels[ind], image_labels[ind]))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
    plt.show()

def main():
    '''
    Read in MNIST data, initialize your model, and train and test your model
    for one epoch. The number of training steps should be your the number of
    batches you run through in a single epoch. You should receive a final accuracy on the testing examples of > 80%.
    :return: None
    '''
    train_inputs, train_labels = get_data('MNIST_data/train-images-idx3-ubyte.gz', 'MNIST_data/train-labels-idx1-ubyte.gz', 60000)
    test_inputs, test_labels = get_data('MNIST_data/t10k-images-idx3-ubyte.gz', 'MNIST_data/t10k-labels-idx1-ubyte.gz', 10000)
    model = Model()
    train(model, train_inputs, train_labels)
    accuracy = test(model, test_inputs, test_labels)
    print("accuracy: " + str((accuracy) * 100) + "%")


    # TODO: load MNIST train and test examples into train_inputs, train_labels, test_inputs, test_labels

    # TODO: Create Model

    # TODO: Train model by calling train() ONCE on all data

    # TODO: Test the accuracy by calling test() after running train()

    # TODO: Visualize the data by using visualize_results()

    pass

if __name__ == '__main__':
    main()
