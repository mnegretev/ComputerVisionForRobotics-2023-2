import cv2
import sys
import random
import numpy
import rospy
import rospkg

NAME = "FULL_NAME"

class NeuralNetwork(object):
    def __init__(self, layers, weights=None, biases=None):
        self.num_layers = len(layers)
        self.layer_sizes = layers
        self.biases = [numpy.random.randn(y, 1) for y in layers[1:]] if biases is None else biases
        self.weights = [numpy.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])] if weights is None else weights

    def feedforward(self, x):
        for b, w in zip(self.biases, self.weights):
            x = sigmoid(numpy.dot(w, x) + b)
        return x

    def feedforward_verbose(self, x):
        y = [x]
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, y[-1]) + b
            a = sigmoid(z)
            y.append(a)
        return y

    def backpropagate(self, x, yt):
        y = self.feedforward_verbose(x)
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        delta = (y[-1] - yt) * sigmoid_prime(y[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = numpy.dot(delta, y[-2].transpose())
        for l in range(2, self.num_layers):
            delta = numpy.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(y[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = numpy.dot(delta, y[-l - 1].transpose())
        return nabla_w, nabla_b

    def update_with_batch(self, batch, eta):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]
        nabla_w = [numpy.zeros(w.shape) for w in self.weights]
        for x, y in batch:
            delta_nabla_w, delta_nabla_b = self.backpropagate(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        m = len(batch)
        self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]

    def get_gradient_mag(self, nabla_w, nabla_b):
        mag_w = sum(numpy.sum(nw * nw) for nw in nabla_w)
        mag_b = sum(numpy.sum(nb * nb) for nb in nabla_b)
        return mag_w + mag_b

    def train_by_SGD(self, training_data, epochs, batch_size, eta):
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k : k + batch_size] for k in range(0, len(training_data), batch_size)]
            for batch in batches:
                self.update_with_batch(batch, eta)
                sys.stdout.write("\rGradient magnitude: %f            " % self.get_gradient_mag(nabla_w, nabla_b))
                sys.stdout.flush()
            print("Epoch: " + str(j))

def sigmoid(z):
    return 1.0 / (1.0 + numpy.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def load_dataset(folder):
    print("Loading dataset from " + folder)
    if not folder.endswith("/"):
        folder += "/"
    training_dataset, training_labels, testing_dataset, testing_labels = [], [], [], []
    for i in range(10):
        f_data = [c / 255.0 for c in open(folder + "data" + str(i), "rb").read(784000)]
        images = [numpy.asarray(f_data[784 * j : 784 * (j + 1)]).reshape([784, 1]) for j in range(1000)]
        label = numpy.asarray([1 if i == j else 0 for j in range(10)]).reshape([10, 1])
        training_dataset += images[0 : len(images) // 2]
        training_labels += [label for j in range(len(images) // 2)]
        testing_dataset += images[len(images) // 2 : len(images)]
        testing_labels += [label for j in range(len(images) // 2)]
    return list(zip(training_dataset, training_labels)), list(zip(testing_dataset, testing_labels))

def main():
    print("PRACTICE 06 - " + NAME)
    rospy.init_node("practice06")
    rospack = rospkg.RosPack()
    dataset_folder = rospack.get_path("config_files") + "/handwritten_digits/"
    epochs = 3
    batch_size = 10
    learning_rate = 3.0

    if rospy.has_param("~epochs"):
        epochs = rospy.get_param("~epochs")
    if rospy.has_param("~batch_size"):
        batch_size = rospy.get_param("~batch_size")
    if rospy.has_param("~learning_rate"):
        learning_rate = rospy.get_param("~learning_rate")

    training_dataset, testing_dataset = load_dataset(dataset_folder)

    try:
        saved_data = numpy.load(dataset_folder + "network.npz", allow_pickle=True)
        layers = [saved_data["w"][0].shape[1]] + [b.shape[0] for b in saved_data["b"]]
        nn = NeuralNetwork(layers, weights=saved_data["w"], biases=saved_data["b"])
        print("Loading data from previously trained model with layers " + str(layers))
    except:
        nn = NeuralNetwork([784, 30, 10])
        pass

    nn.train_by_SGD(training_dataset, epochs, batch_size, learning_rate)
    numpy.savez(dataset_folder + "network", w=nn.weights, b=nn.biases)

    print("\nPress key to test network or ESC to exit...")
    numpy.set_printoptions(formatter={"float_kind": "{:.3f}".format})
    cmd = cv2.waitKey(0)
    while cmd != 27 and not rospy.is_shutdown():
        img, label = testing_dataset[numpy.random.randint(0, 4999)]
        y = nn.feedforward(img).transpose()
        print("\nPerceptron output: " + str(y))
        print("Expected output  : " + str(label.transpose()))
        print("Recognized digit : " + str(numpy.argmax(y)))
        cv2.imshow("Digit", numpy.reshape(numpy.asarray(img, dtype="float32"), (28, 28, 1)))
        cmd = cv2.waitKey(0)


if __name__ == "__main__":
    main()
