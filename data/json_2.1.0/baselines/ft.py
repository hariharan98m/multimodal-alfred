import torch
from dotenv import load_dotenv
import os
import openai
from multiprocessing.pool import ThreadPool as Pool
import tqdm
import pandas as pd

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from utils import Parameters, safe_mkdir, execute, accuracy
import re
from dataset import FTDataset
import json

def finetune(model, dataset_name, modality, num_epochs = 20, save_suffix="_action"):
    checkpoint_dir_path = os.path.join(Parameters.model_path, model.__class__.__name__ + save_suffix,
                                   dataset_name)
    safe_mkdir(checkpoint_dir_path)

    train_dataset = FTDataset('train', modality)
    test_dataset = FTDataset(dataset_name, modality)

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

    train_data_file = os.path.join(checkpoint_dir_path, "train_examples.jsonl")
    test_data_file = os.path.join(checkpoint_dir_path, f"{dataset_name}_examples.jsonl")

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

    test_acc, test_preds, test_accs = accuracy(model, test_dataset.prompts,
                                            true_labels=test_dataset.completions)

    final_csv_path = os.path.join(Parameters.model_path, model.__class__.__name__ + save_suffix, dataset_name,
                              f"{finetune_id}_{model_name}_{test_acc}.csv")

    results = pd.DataFrame({
        'prompt': test_dataset.prompts,
        'true': test_dataset.completions,
        'pred': test_preds,
        'acc': test_accs
    })
    results.to_csv(final_csv_path, index= False)

    return model

def make_predictions():
    dataset_name = 'valid_unseen'
    modality = 'action'
    finetune_id = "ft-oZIrMnZhNesbuvDFamoUIgwG"
    model_name = "ada:ft-carnegie-mellon-university-2023-03-29-00-36-57"
    save_suffix = "_ActionAda"

    model = Finetune()
    model.model = model_name

    test_dataset = FTDataset(dataset_name, modality)
    test_acc, test_preds, test_accs = accuracy(model, test_dataset.prompts,
                                        true_labels=test_dataset.completions)

    checkpoint_dir_path = os.path.join(Parameters.model_path, model.__class__.__name__ + save_suffix,
                                   dataset_name)
    safe_mkdir(checkpoint_dir_path)

    final_csv_path = os.path.join(checkpoint_dir_path, f"{finetune_id}_{model_name}_{test_acc}.csv")

    results = pd.DataFrame({
        'prompt': test_dataset.prompts,
        'true': test_dataset.completions,
        'pred': test_preds,
        'acc': test_accs
    })
    results.to_csv(final_csv_path, index= False)


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
                stop = "(NoOp)"
            )
            # print('logprobs : ', res['choices'][0]['logprobs']['top_logprobs'][0])
            result = response.choices[0].text.strip() + ' (NoOp)'

            return result

        with Pool(25) as p:
            inp_preds = list(tqdm.tqdm(p.imap(predict_example, inps), total=len(inps)))

        return inp_preds

if __name__ == '__main__':
    dataset_name = 'valid_seen'

    # make_predictions()

    # model = Finetune()
    # finetune(model, dataset_name, 'action', num_epochs=20, save_suffix='_ActionAda')

    model = Finetune()
    finetune(model, dataset_name, 'action', num_epochs=20, save_suffix='_ActionAda')
