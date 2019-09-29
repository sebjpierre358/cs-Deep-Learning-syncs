from preprocess import get_data
from assignment import Model, train, test

def main():
    train_inputs, train_labels = get_data('MNIST_data/train-images-idx3-ubyte.gz', 'MNIST_data/train-labels-idx1-ubyte.gz', 60000)
    test_inputs, test_labels = get_data('MNIST_data/t10k-images-idx3-ubyte.gz', 'MNIST_data/t10k-labels-idx1-ubyte.gz', 10000)
    model = Model()
    train(model, train_inputs, train_labels)
    accuracy = test(model, test_inputs, test_labels)
    print("accuracy: " + str((accuracy) * 100) + "%")

main()    
