from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer, get_scheduler
from huggingface_hub import get_full_repo_name
import numpy as np
import evaluate
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from assignment import compute_metrics

# Define label table
label_table = ['O', 'B-corporation', 'I-corporation', 'B-creative-work', 'I-creative-work', 'B-group', 'I-group', 'B-location', 'I-location', 'B-person', 'I-person', 'B-product', 'I-product']

# Load data into HuggingFace Datasets
def load_data(filename):
    data = {"tokens": [], "label_names": [], "labels": []}
    sentence = {"tokens": [], "label_names": [], "labels": []}

    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            line = line.strip()
            if line:
                token, label_name = line.split("\t")
                sentence["tokens"].append(token)
                sentence["label_names"].append(label_name)
                label = label_table.index(label_name)
                sentence["labels"].append(label)
            else:
                if sentence["tokens"]:
                    data["tokens"].append(sentence["tokens"])
                    data["label_names"].append(sentence["label_names"])
                    data["labels"].append(sentence["labels"])
                sentence = {"tokens": [], "label_names": [], "labels": []}
    return Dataset.from_dict(data)

train_set = load_data('wnut17train.conll')
vali_set = load_data('emerging.dev.conll')
test_set = load_data('emerging.test.annotated')

data_dict = {"train": train_set, "vali": vali_set, "test": test_set}
datasets = DatasetDict(data_dict)

# Tokenize datasets
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align_labels(sentences):
    tokenized_inputs = tokenizer(sentences["tokens"], truncation=True, is_split_into_words=True)
    all_labels = sentences["labels"]
    new_labels = [align_labels_with_tokens(labels, tokenized_inputs.word_ids(i)) for i, labels in enumerate(all_labels)]
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels

tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True, remove_columns=datasets["train"].column_names)

# Model and training setup
id2label = {str(i): label for i, label in enumerate(label_table)}
label2id = {v: k for k, v in id2label.items()}
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, id2label=id2label, label2id=label2id)
metric = evaluate.load("seqeval")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# Fine-tuning the model
print("Baseline results:")
trainer.train()
trainer.push_to_hub(commit_message="Training complete")

# Custom Training Loop
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=data_collator, batch_size=8)
vali_dataloader = DataLoader(tokenized_datasets["vali"], collate_fn=data_collator, batch_size=8)
eval_dataloader = DataLoader(tokenized_datasets["test"], collate_fn=data_collator, batch_size=8)

optimizer = AdamW(model.parameters(), lr=2e-5)
num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

print("Results after hyperparameter optimization:")
progress_bar = tqdm(range(num_training_steps)

# Evaluation on the test set
model.eval()
for batch in eval_dataloader:
    with torch.no_grad():
outputs = model(**batch)

predictions = outputs.logits.argmax(dim=-1)
labels = batch["labels"]

true_predictions, true_labels = postprocess(predictions, labels)
metric.add_batch(predictions=true_predictions, references=true_labels)

results = metric.compute()
r_corporation = results["corporation"]
r_creative_work = results["creative-work"]
r_group = results["group"]
r_location = results["location"]
r_person = results["person"]
r_product = results["product"]

macro_f1 = (r_corporation["f1"] + r_creative_work["f1"] + r_group["f1"] + r_location["f1"] + r_person["f1"] + r_product["f1"]) / 6

print(["corporation", r_corporation])
print(["creative_work", r_creative_work])
print(["group", r_group])
print(["location", r_location])
print(["person", r_person])
print(["product", r_product])
print(["macro_f1", macro_f1])

print("Evaluation on test set:", {key: results[f"overall_{key}"] for key in ["precision", "recall", "f1", "accuracy"]})

#%%
