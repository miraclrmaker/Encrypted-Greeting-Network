'''
Author: LiangSiyuan
Date: 2025-12-28 20:32:50
LastEditors: LiangSiyuan
LastEditTime: 2026-01-04 11:51:54
FilePath: /project for fun/encrypt_tool.py
'''
import pandas as pd
import os

def encrypt_message():
    csv_path = 'data.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run generate_data.py first.")
        return

    # 加载映射关系，禁用自动处理空值
    df = pd.read_csv(csv_path, keep_default_na=False)
    mapping = dict(zip(df['input'], df['output']))

    print("--- Encryption Tool ---")
    print("Enter the original message (Case will be preserved):")
    original_text = input("> ")

    encrypted_text = ""
    for char in original_text:
        lower_char = char.lower()
        if lower_char in mapping:
            mapped_char = mapping[lower_char]
            # 如果原字符是大写，加密后的字符也转为大写
            if char.isupper():
                encrypted_text += mapped_char.upper()
            else:
                encrypted_text += mapped_char
        else:
            encrypted_text += char

    print("\nGenerated Encrypted Message (Key):")
    print(encrypted_text)
    print("\nSend this to your friend and ask them to decrypt it using predict.py.")

if __name__ == "__main__":
    encrypt_message()
