import csv
import random
import string

# 定义字符集：26个小写字母 + 4个特殊字符 + 空格
chars = string.ascii_lowercase + "!.?, "
mapping_chars = string.ascii_lowercase + "!.?,"

# 生成随机映射（空格映射为空格，其他字符打乱）
shuffled = list(mapping_chars)
random.seed(42) # 固定随机种子，确保映射可重复
random.shuffle(shuffled)
mapping_dict = dict(zip(mapping_chars, shuffled))
mapping_dict[' '] = ' '

with open('data.csv', 'w', newline='') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL) # 给所有字段加引号，最安全
    writer.writerow(['input', 'output'])
    for char in chars:
        writer.writerow([char, mapping_dict[char]])

print("Dataset data.csv generated.")
print(f"Mapping example: 'a' -> '{mapping_dict['a']}', ' ' -> '{mapping_dict[' ']}'")
