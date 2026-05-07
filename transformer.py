import torch
import os

# os.environ["HF_HUB_DISABLE_XET"] = "1"
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from utils import DatasetManager
from logger_config import setup_logger

logger = setup_logger(__name__)


class TransformerSummarizer:
    def __init__(self, model_name: str = "IlyaGusev/rut5_base_sum_gazeta"):
        """Initialize the transformer model for summarization.

        :param model_name: HuggingFace model name or path to local weights.
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        logger.info(f"Loading tokenizer and model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    # def train(
    #     self,
    #     output_dir: str = "./transformer_model",
    #     epochs: int = 5
    # ) -> None:
    #     """Train (fine-tune) the transformer on the specified dataset.

    #     :param dataset_name: Name of the dataset on HuggingFace Hub.
    #     :param output_dir: Directory to save the fine-tuned model.
    #     :param epochs: Number of training epochs.
    #     """

    #     logger.info(f"Loading dataset {DatasetManager.get_dataset()} for training...")
    #     dataset = DatasetManager.get_dataset()

    #     def preprocess_function(examples):
    #         inputs = examples["text"]
    #         targets = examples["summary"]
    #         model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    #         labels = self.tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    #         model_inputs["labels"] = labels["input_ids"]
    #         return model_inputs

    #     logger.info("Tokenizing dataset...")
    #     tokenized_datasets = dataset.map(
    #         preprocess_function,
    #         batched=True,
    #         remove_columns=dataset["train"].column_names
    #     )

    #     training_args = Seq2SeqTrainingArguments(
    #         output_dir=output_dir,
    #         eval_strategy="epoch",
    #         learning_rate=2e-5,
    #         per_device_train_batch_size=4,
    #         per_device_eval_batch_size=4,
    #         weight_decay=0.01,
    #         save_total_limit=3,
    #         num_train_epochs=epochs,
    #         predict_with_generate=True,
    #         fp16=torch.cuda.is_available(),
    #     )

    #     data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

    #     trainer = Seq2SeqTrainer(
    #         model=self.model,
    #         args=training_args,
    #         train_dataset=tokenized_datasets["train"],
    #         eval_dataset=tokenized_datasets["validation"],
    #         tokenizer=self.tokenizer,
    #         data_collator=data_collator,
    #     )

    #     logger.info("Starting training...")
    #     trainer.train()

    #     logger.info(f"Saving model to {output_dir}")
    #     self.model.save_pretrained(output_dir)
    #     self.tokenizer.save_pretrained(output_dir)
    #     self.model_name = output_dir

    def summarize(self, text: str, limit: int = 300) -> str:
        """Create abstract of given text.

        :param text: initial blob
        :param limit: limit of abstract length in characters
        :return: abstract
        """

        inputs = self.tokenizer(
            [text],
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)


        # Approximate token limit from character limit
        # max_tokens = max(10, limit // 4)

        output_ids = self.model.generate(
            **inputs,
            # max_new_tokens=max_tokens,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # if len(summary) > limit:
        #     summary = summary[:limit].rstrip()

        return summary
