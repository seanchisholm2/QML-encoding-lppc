#!/usr/bin/env python
# coding: utf-8

# ## Amplitude Encoding

# In[157]:


import pennylane as qml
from pennylane import numpy as np # use pennylane's numpy to avoid any errors
import torch


# In[158]:


import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams.update(mpl.rcParamsDefault)
from tqdm import tqdm
import csv
plt.style.use('/Users/seanchisholm/Neutrinos/paper2.mplstyle')


# First Project: Kernel-Based Classifier with Amplitude Encoding

# The aim of this first test run is to practice my ability to create a classifier. From the 2/1 meeting, the goal is to "try to create a kernel-based classifier using amplitude encoding":
# - custom class/function in pennylane that takes space vectors as input $\rightarrow$ inner product or tensor
# - general examples of calculating probability distributions given basis quantum states

# ### Custom Class:

# In[159]:


# Define the quantum device
dev = qml.device("default.qubit", wires=2)

# Define the quantum circuit
@qml.qnode(dev)
def quantum_circuit(params, x):
    # Angle encode the input data onto the qubits
    qml.templates.AngleEmbedding(x, wires=[0, 1])
    # Apply entangling layers to create quantum entanglement
    qml.templates.BasicEntanglerLayers(params, wires=[0, 1])
    # Measure the expectation value of a Pauli-Z operator on the first qubit
    return qml.expval(qml.PauliZ(0))

# Define the inner product kernel function
def inner_product_kernel(x1, x2):
    # Compute the inner product between two vectors
    return np.dot(x1, x2)


# In[160]:


# Define the classifier using the inner product kernel
class KernelClassifier:
    def __init__(self, kernel):
        self.kernel = kernel

    def fit(self, X_train, y_train):
        # Store the training data
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        # Iterate over each test sample
        for x_test in X_test:
            # Compute kernel values between test sample and all training samples
            kernel_values = [self.kernel(x_test, x_train) for x_train in self.X_train]
            # Make prediction by summing the kernel values weighted by training labels
            prediction = np.sign(np.sum(kernel_values * self.y_train))
            predictions.append(prediction)
        return predictions


# In[161]:


# Example usage
X_train = np.array([[1, 0], [0, 1]])  # Training data features
y_train = np.array([1, -1])            # Training data labels
X_test = np.array([[1, 1], [0, 0]])    # Test data features

# Create classifier instance
classifier = KernelClassifier(kernel=inner_product_kernel)
classifier.fit(X_train, y_train)

# Predictions
predictions = classifier.predict(X_test)
print("Predictions:", predictions)


# ***

# In[164]:


def layer(layer_weights):
    for wire, weights in enumerate(layer_weights):
        qml.Rot(*weights, wires=wire)

    for wires in ([0, 1], [1, 2], [2, 3], [3, 0]):
        qml.CNOT(wires)
        
# Define state preparation function:
def state_preparation(x):
    qml.BasisState(x, wires=[0, 1, 2, 3])


# ### Return a Tensor Product:

# In[165]:


# Define the number of qubits
num_qubits = 4

dev = qml.device("default.qubit", wires=4, shots=1000)

@qml.qnode(dev, interface="torch")

# Define the circuit
def circuit(weights, x):
    # Apply state preparation
    state_preparation(x)

    # Apply layers
    for w in weights:
        layer(w)

    # Return PyTorch tensors
    return qml.counts()
    #return [qml.counts(qml.PauliZ(i)) for i in range(num_qubits)]


# Initialize weights
weights = np.random.uniform(0, 2*np.pi, (num_qubits, num_qubits, 3))

# Initialize input data
x = [0, 1, 1, 0]

# Evaluate the circuit
result1 = circuit(weights, x)
#result2 = [torch.Tensor(tensor) for tensor in result1]

# Print the numerical values directly
print(result1)
print("Expectation values:", result1)


# In[166]:


def tensor_product(tensors):
    tensor_product = qml.numpy.tensor(tensors)
    return tensor_product

result_tensor_product = tensor_product(result1)
print("Tensor Product:")
print(result_tensor_product)


# ***

# In[1]:


def tensor_product_from_values(*values):
    n = len(values)
    dim = 2 ** n
    tensor = np.zeros(dim, dtype=complex)
    tensor[0:len(values)] = values
    norm = np.linalg.norm(tensor)
    tensor /= norm
    dev = qml.device("default.qubit", wires=n)
    
    @qml.qnode(dev)
    def circuit():
        qml.QubitStateVector(tensor, wires=list(range(n)))
        return qml.state()
    
    return circuit()


# In[171]:


def plot_tensor_product(tensor):
    probs = np.abs(list(tensor.values()))**2
    n = int(np.log2(len(probs)))

    basis_states = [format(i, '0' + str(n) + 'b') for i in range(2**n)]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(2**n), probs, tick_label=basis_states)
    ax.set_xlabel('Basis State')
    ax.set_ylabel('Probability')
    ax.set_title('Tensor Product State')
    plt.xticks(rotation=90)
    plt.grid()
    plt.tight_layout()
    plt.show()

#tensor_prod = np.array(values)
result2 = {key: value/1000 for key, value in result1.items()}

plot_tensor_product(result2)
print(type(result2))


# In[ ]:


# Compute kernel matrices
kernel_matrix_train = compute_kernel_matrix(prob_dist_train)
kernel_matrix_test = compute_kernel_matrix(prob_dist_test)

# Use kernel matrix in SVM
clf = svm.SVC(kernel='precomputed')
clf.fit(kernel_matrix_train, y_train)

# Predictions
y_pred_train = clf.predict(kernel_matrix_train)
y_pred_test = clf.predict(kernel_matrix_test)

# Evaluate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# ***

# ### Training via Pennylane

# #### Optimization of Cost Function

# In[95]:


# standard square loss that measures the distance between target labels and model predictions:

def square_loss(labels, predictions):
    return np.mean((labels - qml.math.stack(predictions)) ** 2)

# accuracy - the proportion of predictions that agree with a set of target labels
def accuracy(labels, predictions):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc

# cost depends on the data - features and labels considered in the iteration of the optimization routine
def cost(weights, X, Y):
    predictions = [circuit(weights, x) for x in X]
    return square_loss(Y, predictions)


# #### Optimization

# In[96]:


data = np.loadtxt("practice_datasets/seeds_dataset.txt", dtype=float)
print(data.shape)
X = np.array(data[:, :-1])
Y = np.array(data[:, -1])


# check data:
for x,y in zip(X, Y):
    print(f"x = {x}, y = {y}")
    


# In[97]:


np.random.seed(0)
num_qubits = 4
num_layers = 2
weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)

print("Weights:", weights_init)


# In[ ]:




