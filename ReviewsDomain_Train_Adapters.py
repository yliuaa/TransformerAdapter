import numpy as np
from datasets import load_dataset
from transformers import RobertaTokenizer
from transformers import RobertaConfig, RobertaModelWithHeads, AutoAdapterModel
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from sklearn.metrics import f1_score
import argparse
from datasets import ClassLabel

def eval_metric(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  macro_f1 = f1_score(p.label_ids, preds, average='macro')
  return {"acc": (preds == p.label_ids).mean(), "macro-f1": macro_f1.mean()}


def get_task_dataset(task_name, max_length):
  tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

  if task_name == "imdb":
    def encode_batch(batch):
      return tokenizer(batch["text"], max_length=max_length, truncation=True, padding="max_length")

    dataset = load_dataset("imdb")

  elif task_name == "helpfulness":
    label_converter = ClassLabel(num_classes=2, names=['helpful', 'unhelpful'])

    def encode_batch(batch):
      tokens = tokenizer(batch["text"], max_length=512, truncation=True, padding="max_length")
      tokens["label"] = label_converter.str2int(batch["label"])
      return tokens

    dataset = load_dataset("vannacute/AmazonReviewHelpfulness")

  else:
    ValueError("%s dataset is not implemented yet" % task_name)

  # Transform datset to fit pytorch format
  dataset = dataset.map(encode_batch, batched=True)
  dataset = dataset.rename_column("label", "labels")
  dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

  return dataset


def build_model(task_name, num_labels, adapter_config=None, checkpoint=None):
  if checkpoint:
    model = AutoAdapterModel.from_pretrained(checkpoint)
    model.load_adapter(checkpoint)

  else:
    config = RobertaConfig.from_pretrained(
      "roberta-base",
      num_labels=num_labels,
    )

    # load model
    model = RobertaModelWithHeads.from_pretrained(
      "roberta-base",
      config=config,
    )

    # Add a new adapter
    model.add_adapter(task_name, config=adapter_config)

    # Add a matching classification head
    model.add_classification_head(
      task_name,
      num_labels=num_labels,
      id2label={ 0: "dislike", 1: "like"}
    )

  # Activate the adapter
  model.train_adapter(task_name)

  return model


def train(task_name,
          model,
          output_dir,
          lr,
          num_epochs,
          batch_size,
          seed,
          eval_only=False):

  training_args = TrainingArguments(
    learning_rate=lr,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_steps=200,
    output_dir=output_dir,
    overwrite_output_dir=True,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    seed=seed,
  )

  trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=eval_metric,
  )

  if not eval_only:
    # start training
    train_output = trainer.train()
    print(train_output)

    # save model
    model.save_pretrained(output_dir)
    # save adapter
    model.save_adapter(output_dir, task_name)

  # evaluation
  eval_output = trainer.evaluate()
  print(eval_output)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--task',       required=True, help='Task name: imdb or helpfullness')
  parser.add_argument('--output',     required=True, help='Output path of trained model')

  parser.add_argument('--task_cls',   default=2,                help='Task class_number')
  parser.add_argument('--num_epochs', default=10,   type=int,   help='Number of epochs')

  parser.add_argument('--seed',       default=42,   type=int,   help='Random seed')
  parser.add_argument('--eval',       action='store_true',      help='Evaluation only?')
  parser.add_argument('--ckpt',       default=None,             help='Input path of previous model')

  ##### Learning Rate: tried 1e-3, 5e-4, 1e-4
  parser.add_argument('--lr',         default=1e-4, type=float, help='Learning rate')

  ##### Batch Size: tried 32, 64, 128
  parser.add_argument('--batch_size', default=32,   type=int,   help='Batch size')

  ##### Max Length: tried 128, 256, 512, 1024
  parser.add_argument('--max_length', default=512,  type=int,   help='Sequence max length')

  ##### Adapter Configurations: tried Pfeiffer, Houlsby, Parallel
  parser.add_argument('--adapter_config', default=None,         help='Adapter Config')

  args = parser.parse_args()

  # get dataset
  dataset = get_task_dataset(args.task, args.max_length)

  # build model
  model = build_model(args.task, args.task_cls, args.adapter_config, args.ckpt)

  # train
  train(task_name=args.task,
        model=model,
        output_dir=args.output,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        eval_only=args.eval)

