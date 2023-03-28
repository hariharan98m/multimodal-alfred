import torch
from dotenv import load_dotenv
import os
import openai
from multiprocessing.pool import ThreadPool as Pool
import tqdm

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from utils import Parameters, safe_mkdir, execute, accuracy
import re
from dataset import FTDataset
import json

def finetune(model, dataset_name, test_dataset_name, modality, num_epochs = 20, save_suffix="_action"):
    checkpoint_path = os.path.join(Parameters.model_path, model.__class__.__name__ + save_suffix,
                                   dataset_name)
    final_path = os.path.join(Parameters.model_path, model.__class__.__name__ + save_suffix, dataset_name,
                              f"final_chkpt.json")
    safe_mkdir(checkpoint_path)

    train_dataset = FTDataset(dataset_name, modality)
    test_dataset = FTDataset(test_dataset_name, modality)

    # frame train and test examples.
    train_examples = []
    for data_text, label in train_dataset:
        train_examples.append(json.dumps({
            'prompt': data_text,
            'completion': label
        }) + '\n')
    # create test examples.
    test_examples = []
    for data_text, label in test_dataset:
        test_examples.append(json.dumps({
            'prompt': data_text,
            'completion': label
        }) + '\n')

    train_data_file = os.path.join(Parameters.model_path, model.__class__.__name__ + save_suffix, dataset_name,
                              "train_examples.jsonl")
    test_data_file = os.path.join(Parameters.model_path, model.__class__.__name__ + save_suffix, dataset_name,
                                   "test_examples.jsonl")

    # dump examples into these files.
    with open(train_data_file, 'w') as f:
        f.writelines(train_examples)
    with open(test_data_file, 'w') as f:
        f.writelines(test_examples)

    start_cmd = f'''export OPENAI_API_KEY={openai.api_key} && openai api fine_tunes.create -m ada \
                -t {train_data_file} \
                -v {test_data_file} \
                --n_epochs {num_epochs}'''

    for line in execute(start_cmd):
        print(line)
        m = re.search('ft-.*', line)
        if m and 'Created fine-tune:' in line:
            finetune_id = m.group()
            break

    follow_cmd = f'''export OPENAI_API_KEY={openai.api_key} && \
                openai api fine_tunes.follow -i {finetune_id}'''

    done = False

    while True:
        for line in execute(follow_cmd):
            print(line, end='')
            m = re.search('ada\S+', line)
            if m:
                model_name = m.group()
                done = True
                break
        if done:
            break

    model.model = model_name

    train_acc, train_preds, train_accs = accuracy(model, train_dataset.prompts,
                                      true_labels= train_dataset.labels)

    test_acc, test_preds, test_accs = accuracy(model, test_dataset.data_texts,
                                            true_labels=test_dataset.labels)

    stats = {
        'finetune_id': finetune_id,
        'model_params': {k: v for k, v in vars(model).items() if not k.startswith('_')},
        'train_err': train_acc,
        'train_preds': train_preds,
        'test_err': test_acc,
        'test_preds': test_preds
    }

    print(f'train acc = {train_acc:.4f}, test acc = {test_acc:.4f}')
    json.dump(stats, open(final_path, 'w'), indent=4)

    return model


class Finetune(torch.nn.Module):
    def __init__(self):
        super(Finetune, self).__init__()
        self.model = None

    def forward(self, inps):
        '''
            inps: list of data texts.
            returns : predicted classes.
        '''

        assert self.model != None

        def predict_example(prompt):
            response = openai.Completion.create(
                model=self.model,
                prompt= prompt,
                temperature=0.0,
                max_tokens=200,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            # print('logprobs : ', res['choices'][0]['logprobs']['top_logprobs'][0])
            result = response.choices[0].text.strip()

            return result

        with Pool(25) as p:
            inp_preds = list(tqdm.tqdm(p.imap(predict_example, inps), total=len(inps)))

        return inp_preds

if __name__ == '__main__':
    train_dataset_name = 'train'
    test_dataset_name = 'valid_unseen'

    model = Finetune()
    finetune(model, train_dataset_name, test_dataset_name, 'action', num_epochs=20, save_suffix='_ActionAda')
