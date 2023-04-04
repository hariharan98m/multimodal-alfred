import torch
from dotenv import load_dotenv
import os
import openai
from multiprocessing.pool import ThreadPool as Pool
import tqdm
from dataset import PromptingDataset
from utils import accuracy, safe_mkdir, Parameters
import pandas as pd
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class Prompting(torch.nn.Module):
    def __init__(self, modality, model, action_vocab, object_vocab):
        super(Prompting, self).__init__()
        self.model = model
        self.modality = modality
        self.action_vocab= action_vocab
        self.object_vocab= object_vocab

    def forward(self, inps):
        '''
            inps: list of data texts.
            returns : predicted completions.
        '''
        def predict_example(inp):
            instruction, support_examples = inp

            support = []
            for (i, a) in support_examples:
                support.append(f'''I: {i}
A: {a}
''')
            support_string = '\n'.join(support)
            if self.modality == 'language':
                prompt = f'''We have a set of expert demonstrations from the ALFRED (A Benchmark for Interpreting Grounded Instructions for Everyday Tasks) dataset.
Each demonstration is a set of high-level language instructions that were grounded in the scence and a sequence of actions that were executed by the agent to achieve the goal.

{support_string}

We want to predict the action sequence needed to accomplish the goal given the instruction. Action vocabulary: {self.action_vocab}, object vocabulary: {self.object_vocab}.
I: {instruction}
A: '''
            else:
                goal = instruction
                prompt = f'''We have a set of expert demonstrations from the ALFRED (A Benchmark for Interpreting Grounded Instructions for Everyday Tasks) dataset.
Each demonstration is a sequence of actions that were executed by the agent to achieve a given goal.

{support_string}

We want to predict the action sequence needed to accomplish the below task. Action vocabulary: {self.action_vocab}, object vocabulary: {self.object_vocab}.
{goal}
A: '''

            if self.model == 'gpt-4':
                while True:
                    try:
                        response = openai.ChatCompletion.create(
                            model=self.model,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )
                        inp_pred = response.choices[0].message.content.strip()
                        break
                    except:
                        time.sleep(5)
                        continue
            else:
                response = openai.Completion.create(
                    model = self.model,
                    prompt = prompt,
                    temperature = 0.0,
                    max_tokens = 200,
                    # top_k=50,
                    # top_p = 0.95,
                    stop = "(NoOp)"
                )
                inp_pred = response.choices[0].text.strip() + ' (NoOp)'

            return inp_pred

        with Pool(20) as pool:
            inp_preds = list(tqdm.tqdm(pool.imap(predict_example, inps), total=len(inps)))

        return inp_preds

def experiment_prompting(modality, dataset_name, model_name = 'ActionPrompting'):
    dataset = PromptingDataset(dataset_name, modality = modality)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size= 10, shuffle= False,
                                             num_workers= 0)
    prompting = Prompting(model = 'gpt-4',
                          modality = modality,
                          action_vocab= dataset.action_vocab,
                          object_vocab= dataset.object_vocab)
    inps = []
    true_labels = []
    trial_ids = []
    for i, ((ins_batch, a_batch), train_examples_batch, trial_ids_batch) in enumerate(dataloader):
        for j in range(len(ins_batch)):
            # create a list of (instruction, action) pairs for the closest examples.
            close_examples = []
            for example in train_examples_batch:
                close_examples.append((example[0][j], example[1][j]))
            inps.append((ins_batch[j], close_examples))
            true_labels.append(a_batch[j])
            trial_ids.append(trial_ids_batch[j])

    # inp_preds = prompting(inps, action_vocab= dataset.action_vocab,
    #                           object_vocab= dataset.object_vocab)
    acc, inp_preds, accs = accuracy(prompting, inps, true_labels)
    print('accuracy: {:.3f}'.format(acc))

    results = pd.DataFrame({
        'trial_ids': trial_ids,
        'inps': [inp[0] for inp in inps],
        'labels': true_labels,
        'preds': inp_preds,
        'accs': accs,
    })

    final_dir_path = os.path.join(Parameters.model_path, model_name)
    safe_mkdir(final_dir_path)

    final_path = os.path.join(final_dir_path, f'{dataset_name}_{acc}.csv')
    results.to_csv(final_path, index= False)


if __name__ == '__main__':
    # experiment_language_prompting()
    modality = 'action'
    experiment_prompting(modality = modality, dataset_name = 'valid_unseen', model_name = modality.capitalize() + 'Prompting')