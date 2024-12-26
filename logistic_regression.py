from EDF import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define constants hyperparamters
CLASS1_SIZE = 500
CLASS2_SIZE = 500
N_FEATURES = 2
N_OUTPUT = 1
epochs = 100
learning_rate = 0.001
TEST_SIZE = 0.25
BATCH_SIZE = 1    # To perform SGD choose BATCH_SIZE == 1

# Define the means and covariances of the two components
MEAN1 = np.array([1, 2])
COV1 = np.array([[1, 0], [0, 1]])
MEAN2 = np.array([1, -2])
COV2 = np.array([[1, 0], [0, 1]])

# Generate random points from the two components
X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)

# Combine the points and generate labels
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

# Plot the generated data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Generated Data')
plt.show()

# Split data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Model parameters
n_features = X_train.shape[1]
n_output = 1

# Initialize weights and biases
W0 = np.zeros((1 , n_output))
W = np.random.randn(n_features, n_output) * 0.1


# Create nodes
# Input vector (stacked features)
x_node = Input()
y_node = Input()

# Parameter matrix (stacked weights)
w_node = Parameter(W)
w0_node = Parameter(W0)

# Build computation graph
u_node = Linear(x_node,w_node,w0_node)
sigmoid = Sigmoid(u_node)
loss = BCE(y_node, sigmoid)

# Create graph outside the training loop
graph = [x_node,w_node,w0_node,u_node,sigmoid,loss]
trainable = [w0_node,w_node]

# Forward and Backward Pass
def forward_pass(graph):
    for n in graph:
        n.forward()

def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()

# SGD Update
def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
          t.value -= learning_rate * np.mean(t.gradients[t], axis=0).reshape(-1, 1)

# store training losses for batch sizes
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
losses_for_batch_sizes = {}

# 2x4 subplots for contour plots
fig1, axes1 = plt.subplots(2, 4, figsize=(15, 8))

# Loop over batch sizes
for index, BATCH_SIZE in enumerate(batch_sizes):
    Epoch_arr = []
    loss_arr = []
    # Reset weights for each batch size
    W0 = np.zeros((1 , n_output))
    W = np.random.randn(n_features, n_output) * 0.1
    w0_node.value = W0
    w_node.value = W
    
    # Training loop
    for epoch in range(epochs):
        loss_value = 0
        # Training for each batch
        for i in range(0, X_train.shape[0], BATCH_SIZE):
            x_node.value = X_train[i:i + BATCH_SIZE]
            y_node.value = y_train.reshape(-1, 1)[i:i + BATCH_SIZE]     
            forward_pass(graph)
            backward_pass(graph)
            sgd_update(trainable, learning_rate)
            loss_value += loss.value
        
        # Calculate average loss for each epoch (for plotting)
        loss_value /= (X_train.shape[0] / BATCH_SIZE)
        Epoch_arr.append(epoch + 1)
        loss_arr.append(loss_value)
        print(f"Epoch {epoch + 1}, Batch Size {BATCH_SIZE}, Loss: {loss_value}")
    
    # Evaluate the model
    correct_predictions = 0
    x_node.value = X_test
    y_node.value = y_test
    forward_pass(graph)
    for i in range(X_test.shape[0]):
        if (sigmoid.value[i]>=0.5) == y_test[i]:
            correct_predictions += 1
    accuracy = correct_predictions / X_test.shape[0]
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    
    # Prepare for contour plot
    x_min, x_max = X_train[:, 0].min(), X_train[:, 0].max()
    y_min, y_max = X_train[:, 1].min(), X_train[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
    x_node.value = np.array(list(zip(xx.ravel(),yy.ravel())))
    y_node.value = np.zeros((xx.ravel().shape[0], 1))
    forward_pass(graph)
    Z = np.array(sigmoid.value).reshape(xx.shape)
    
    # Plot the contour
    ax = axes1[index // 4, index%4] 
    ax.contourf(xx, yy, Z, alpha=0.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'Batch Size = {BATCH_SIZE}\nAccuracy: {accuracy * 100:.2f}%')

    # Store the losses for loss curves 
    losses_for_batch_sizes[BATCH_SIZE] = loss_arr

# Show the contour plots
plt.tight_layout()
plt.show()

# 2x4 subplots for Training loss plots
fig2, axes2 = plt.subplots(2, 4, figsize=(15, 8))

# Plot the training loss plots for all batch sizes
for index, BATCH_SIZE in enumerate(batch_sizes):
    ax = axes2[index // 4, index % 4]  # Get the correct subplot axis
    ax.plot(Epoch_arr, losses_for_batch_sizes[BATCH_SIZE], label=f"Batch Size {BATCH_SIZE}")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Training Loss')
    ax.set_title(f'Training Loss\nBatch Size = {BATCH_SIZE}')
    ax.grid(True)
    ax.legend()

# Show the training loss plots
plt.tight_layout()
plt.show()


