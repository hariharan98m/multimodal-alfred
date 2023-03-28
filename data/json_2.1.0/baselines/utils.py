import tiktoken
from collections import defaultdict
from pathlib import Path
from numpy.random import choice
import json
import os
import subprocess

encoding = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Parameters:
    model_path = 'Users/hariharan/hari_works/alfred/data/json_2.1.0/baselines/models'
    datasets_path = 'Users/hariharan/hari_works/alfred/data/json_2.1.0'


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, shell = True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

def accuracy(model, inps, true_labels):
    correct = 0
    total = 0
    accs = []

    inp_preds = model(inps)

    for inp_pred, true_label in zip(inp_preds, true_labels):
        preds = inp_pred.split('->')
        labels = true_label.split('->')
        sample_correct = 0
        for l in labels:
            if l in preds:
                sample_correct +=1
                correct += 1
        total += len(labels)
        accs.append(sample_correct/len(labels))
    return correct / total, accs

def generate_instructions_actions(data_path, modality = 'language'):
    data = defaultdict(list)
    task_type2keys = defaultdict(list)
    object_vocab = set()
    action_vocab = set()

    examples = []
    for path in Path(data_path).iterdir():
        if path.is_dir():
            task_type, obj, parent, recep, _ = path.name.split('-')
            for json_file in path.rglob('*.json'):
                json_object = json.load(open(json_file, 'r'))

                plan = json_object['plan']['high_pddl']
                A = []
                for item in plan:
                    act = item['discrete_action']['action']
                    args = item['discrete_action']['args']
                    object_vocab.update(args)
                    action_vocab.add(act)
                    A.append('(' + ', '.join([act] + args) + ')')
                A = '-> '.join(A)
                if not A.endswith('(NoOp)'):
                    A += '-> (NoOp)'

                demonstration = choice(json_object['turk_annotations']['anns'])
                I = f"Goal: {demonstration['task_desc']}."
                if modality == 'language':
                    I += " Steps: {'-> '.join(demonstration['high_descs'])}"

                key = (task_type, obj, parent, recep)
                data[key].append((I, A))
                task_type2keys[task_type].append(key)
                examples.append((I, A, key))

    object_vocab -= {''}
    return data, task_type2keys, examples, object_vocab, action_vocab

data_root = "/Users/hariharan/hari_works/alfred/data/json_2.1.0"
if __name__ == '__main__':
    train_data, task_type2keys, _, object_vocab, action_vocab = generate_instructions_actions(data_path = os.path.join(data_root, 'train'))
    print(action_vocab)
    print(object_vocab)