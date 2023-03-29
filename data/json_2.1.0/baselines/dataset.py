import torch
from numpy.random import choice
from utils import generate_instructions_actions, num_tokens_from_string
import json
import random
import os

data_root =  "/Users/hariharan/hari_works/alfred/data/json_2.1.0"

class PromptingDataset(torch.utils.data.Dataset):
    def __init__(self, dir_name, modality, num_examples = 5):
        '''
            data_path : path to the jsons directory.
        '''
        super(PromptingDataset, self).__init__()
        self.modality = modality
        self.train_data, self.task_type2keys, _, self.object_vocab, self.action_vocab = generate_instructions_actions(data_path = os.path.join(data_root, 'train'), modality = self.modality)
        _, _, self.examples, _, _ = generate_instructions_actions(data_path = os.path.join(data_root, dir_name), modality = self.modality)
        self.num_examples = num_examples
        self.length = len(self.examples)

    def __getitem__(self, item):
        '''
            Returns a tuple of (instruction, action)
        '''
        i, a, key = self.examples[item]
        if len(self.train_data[key]) > 0 :
            closest_example_indices = choice(range(len(self.train_data[key])), self.num_examples)
            some_train_examples = [self.train_data[key][i] for i in closest_example_indices]
        else:
            task_type, _, _, _ = key
            random.shuffle(self.task_type2keys[task_type])
            closest_keys = self.task_type2keys[task_type][:self.num_examples]

            some_train_examples = []
            for key in closest_keys:
                rand_idx = random.randint(0, len(self.train_data[key])-1)
                some_train_examples.append(self.train_data[key][rand_idx])

        return (i, a), some_train_examples

    def __len__(self):
        return self.length


class FTDataset(torch.utils.data.Dataset):
    def __init__(self, dir_name, modality):
        '''
            data_path : path to the jsons directory.
        '''
        super(FTDataset, self).__init__()
        self.modality = modality
        self.data, _, self.examples, self.object_vocab, self.action_vocab = generate_instructions_actions(data_path = os.path.join(data_root, dir_name), modality = self.modality)

        self.datalist = []
        self.prompts = []
        self.completions = []
        for instructions, actions, _ in self.examples:
            if dir_name == 'train':
                acts = actions.split('->')
                for i in range(5):
                    label = '-> '.join(acts[i:])
                    prompt = instructions + ' | Actions: ' + '-> '.join(acts[:i]) + '-> '
                    self.prompts.append(prompt)
                    self.completions.append(label)
                    self.datalist.append((prompt, label))
            else:
                prompt = instructions + ' | Actions: '
                self.prompts.append(prompt)
                self.completions.append(actions)
                self.datalist.append((prompt, actions))

        self.length = len(self.datalist)

    def __getitem__(self, item):
        '''
            Returns a tuple of (instruction + action, remaining actions)
        '''
        prompt, label = self.datalist[item]
        return prompt, label

    def __len__(self):
        return self.length

if __name__ == '__main__':
    dataset = PromptingDataset('valid_unseen')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 10, shuffle = True)

    for i, ((ins, a), train_examples) in enumerate(dataset):
        print('I:', ins)
        print('A:', a)
        print('Train examples:')
        for train_ins, a in train_examples:
            print(train_ins, a)
        print('-'* 50)
        if i == 1:
            break

    # dataset = LanguageFTDataset('valid_unseen')
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size = 10, shuffle = True)

    # for i, (prompt, label) in enumerate(dataset):
    #     print('prompt:', prompt)
    #     print('label:', label)
    #     print('-'* 50)
    #     if i == 5:
    #         break