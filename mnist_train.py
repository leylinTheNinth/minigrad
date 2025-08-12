import numpy as np
from mytorch.nn.sequential import Sequential
from mytorch.nn.linear import Linear
from mytorch.nn.loss import CrossEntropyLoss
from mytorch.nn.activations import ReLU
from mytorch.optim.sgd import SGD
import mytorch.nn.functional as F

from mytorch.tensor import Tensor


BATCH_SIZE = 100

def mnist(train_x, train_y, val_x, val_y):
    my_model = Sequential(
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 10)
    )
    my_optimizer = SGD(my_model.parameters(), momentum= 0.9, lr= 0.1)
    my_criterion = CrossEntropyLoss()
    val_accuracies = train(my_model,
                           my_optimizer,
                           my_criterion,
                           train_x, train_y, val_x, val_y)
    return val_accuracies


def train(model, optimizer, criterion, train_x, train_y, val_x, val_y, num_epochs=6):
    val_accuracies = []
    model.train()

    for epoch in range(num_epochs):
        shuffled_x, shuffled_y = shuffle_train_data(train_x, train_y)
        batches = split_data_into_batches(shuffled_x, shuffled_y)
        for i, (batch_data, batch_label) in enumerate(batches):
            optimizer.zero_grad()
            input_data, label = Tensor(batch_data), Tensor(batch_label)
            logits = model(input_data)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            if i%100 == 0:
                accuracy = validate(model, val_x, val_y)
                print(f"Val accuracy at step i = {i} of epoch {epoch}: {accuracy}")
                val_accuracies.append(accuracy)
                model.train()
    return val_accuracies

def validate(model, val_x, val_y):
    model.eval()
    batches = split_data_into_batches(val_x, val_y)
    num_correct = 0
    for (batch_data, batch_label) in batches:
        input_data = Tensor(batch_data)
        logits = model(input_data)
        predicted_labels = np.argmax(logits.data, axis= 1)
        num_correct += np.sum(predicted_labels == batch_label)
    accuracy = num_correct / len(val_y) 
    return accuracy

def shuffle_train_data(train_x, train_y):
    indices = np.random.permutation(len(train_y))
    shuffled_x = train_x[indices]
    shuffled_y = train_y[indices]
    
    return shuffled_x, shuffled_y


def split_data_into_batches(data_x, data_y, batch_size=BATCH_SIZE):
    num_samples = len(data_y)
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        batch_x = data_x[start_idx:end_idx]
        batch_y = data_y[start_idx:end_idx]
        
        batches.append((batch_x, batch_y))
    
    return batches
