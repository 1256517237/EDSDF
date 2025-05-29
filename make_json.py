import os
import json
import argparse
import random

current_directory = os.getcwd()

def generate_split(data_path):
    li = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            li.append(file_path)
    
    random.shuffle(li)

    print(len(li))
    with open('train.json', 'w') as fw:
        entry = json.dumps(li, indent=0)
        fw.write(entry)
        fw.write('\n')

if __name__ == '__main__':
    generate_split('dataset/train/images')
