import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# device config
if (torch.cuda.is_available()):
	print("Using CUDA")
	device = torch.device('cuda')
else:
	print("Using CPU")
	device = torch.device('cpu')

# File config
FILE = "ModelParameters.pth"

# hyper parameters
input_size = 3
hidden_size = 5
output_size = 1
num_epochs = 5
batch_size = 20
learning_rate = 0.1

# features parameters
starting_hour = 0.0  # 6 AM
ending_hour = 24.0 # 8 PM
max_temp = 50.0
max_power = 50.0 # 50 watt panel


# Data importing
csvData = pd.read_csv('dataReal.csv')
#print(csvData)
Data = torch.tensor(csvData.values)
#print(Data)
data_size = Data.numel()/4

# Data normalization
Data[:,0] = (Data[:,0]-starting_hour) / (ending_hour-starting_hour) # Time
Data[:,1] = Data[:,1] / max_temp # Temp
Data[:,2] = Data[:,2] / 100.0 # Humidity
Data[:,3] = Data[:,3] / max_power # Power

# Split into Test and Train (80:20)
train_size = int(0.8 * data_size)
test_size = int(data_size - train_size)
#print (train_size, " ",test_size)
train_set, test_set = torch.utils.data.random_split(Data,[train_size,test_size])

train_loader = torch.utils.data.DataLoader(dataset= train_set, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset= test_set, batch_size = batch_size, shuffle = False)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size,hidden_size,output_size).to(device)

#load saved parameters
#model.load_state_dict(torch.load(FILE))



#loss and optimization
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#training loop
steps = len(train_loader)
for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):
        inputs = batch[:,0:3].float().to(device)
        labels = batch[:,3].float().to(device)

        #forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'epoch {epoch + 1} / {num_epochs}, step {i + 1}/{steps} , loss = {loss.item()}')
        if (i+1) % (train_size / 10) == 0:
            print (f'epoch {epoch + 1} / {num_epochs}, step {i+1}/{steps} , loss = {loss.item()}')


#testing loop
with torch.no_grad():
    MSE = 0
    testing_samples = 0

    for i, batch in enumerate(test_loader):
        inputs = batch[:, 0:3].float().to(device)
        labels = batch[:, 3].float().to(device)


        outputs = model(inputs)
        outputs = np.squeeze(outputs)

        error = outputs-labels
        MSE += torch.dot(error,error).item()
        testing_samples += labels.shape[0]


    MSE /= testing_samples

    print ("Mean Square Error (MSE) = ", MSE)

#saving model parameters
torch.save(model.state_dict(), FILE)

