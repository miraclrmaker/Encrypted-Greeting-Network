'''
Author: LiangSiyuan
Date: 2025-12-28 20:30:43
LastEditors: LiangSiyuan
LastEditTime: 2025-12-28 21:06:39
FilePath: /project for fun/train.py
'''
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import string

chars = string.ascii_lowercase + "!.?, "
char_to_idx = {char: i for i, char in enumerate(chars)}
idx_to_char = {i: char for i, char in enumerate(chars)}
num_chars = len(chars)

class BirthdayNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BirthdayNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def one_hot(char):
    vec = torch.zeros(num_chars)
    vec[char_to_idx[char]] = 1
    return vec

def train():
    df = pd.read_csv('data.csv', keep_default_na=False)
    inputs = torch.stack([one_hot(c) for c in df['output']])
    targets = torch.tensor([char_to_idx[c] for c in df['input']])
    model = BirthdayNet(num_chars, 64, num_chars)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 1500
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as model.pth")

if __name__ == "__main__":
    train()
