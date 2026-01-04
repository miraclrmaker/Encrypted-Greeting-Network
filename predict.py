'''
Author: LiangSiyuan
Date: 2025-12-28 20:31:24
LastEditors: LiangSiyuan
LastEditTime: 2025-12-28 21:07:58
FilePath: /project for fun/predict.py
'''
import torch
import torch.nn as nn
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
    if char in char_to_idx:
        vec[char_to_idx[char]] = 1
    return vec

def predict(text, model, device):
    model.eval()
    result = ""
    with torch.no_grad():
        for char in text:
            lower_char = char.lower()
            if lower_char in char_to_idx:
                input_vec = one_hot(lower_char).to(device)
                output = model(input_vec)
                pred_idx = torch.argmax(output).item()
                pred_char = idx_to_char[pred_idx]
                
                # 保持原输入的大小写状态
                if char.isupper():
                    result += pred_char.upper()
                else:
                    result += pred_char
            else:
                result += char # 不在字符集内的字符保持原样
    return result

if __name__ == "__main__":
    # 自动检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # (input_size 现在是 31)
    model = BirthdayNet(num_chars, 64, num_chars).to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=True))
    
    print(f"--- Birthday Message Decryption (Running on {device}) ---")
    print("Please enter the encrypted message (gibberish):")
    encrypted_text = input("> ")
    
    decrypted_text = predict(encrypted_text, model, device)
    print("\nDecryption Result:")
    print(decrypted_text)
