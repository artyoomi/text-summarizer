import random
import cli
import os

from baseline import TextRankSummarizer
from transformer import TransformerSummarizer
from metrics import compute_rouge
from logger_config import setup_logger
from utils import DatasetManager

logger = setup_logger(__name__)


if __name__ == "__main__":
    args = cli.parse_args()

    summarizer = None
    match args.method:
        case cli.Method.TEXTRANK:
            summarizer = TextRankSummarizer()
        case cli.Method.T5:
            summarizer = TransformerSummarizer()

    if args.from_dataset > 0:
        refs: list[str] = []
        preds: list[str] = []

        # os.environ["HF_HUB_OFFLINE"] = "1"
        dataset = DatasetManager.get_dataset()

        if (args.metrics):
            references  = []
            predictions = []

        for i in random.sample(range(0, len(dataset['train'])), args.from_dataset):
            logger.info(f"Dataset instance no. {i+1}")
            logger.info(f"\n\ttext: {dataset['train'][i]['text']}")
            logger.info(f"\n\treference summary: {dataset['train'][i]['summary']}")

            summary = summarizer.summarize(dataset['train'][i]['text'])
            logger.info(f"\n\tinferenced summary: {summary}")

            if args.metrics:
                references.append(dataset['train'][i]['summary'])
                predictions.append(summary)

        if args.metrics:
            rouge = compute_rouge(predictions, references)
            lines = ["\nROUGE (min / mean / max over samples)"]
            for name in ("rouge1", "rouge2", "rougeL"):
                m = rouge[name]
                lines.append(f"\t{name}:")
                lines.append(
                    f"\t  precision: min={m.precision[0]:.4f} mean={m.precision[1]:.4f} max={m.precision[2]:.4f}"
                )
                lines.append(
                    f"\t  recall:    min={m.recall[0]:.4f} mean={m.recall[1]:.4f} max={m.recall[2]:.4f}"
                )
                lines.append(
                    f"\t  f1:        min={m.f1[0]:.4f} mean={m.f1[1]:.4f} max={m.f1[2]:.4f}"
                )
            logger.info("\n".join(lines))
    else:
        input = input()
        summary = summarizer.summarize(input)
        logger.info(f"\n\tinferenced summary: {summary}")

    # summarizer = TextRankSummarizer()

    # ds = load_dataset("IlyaGusev/gazeta")
    # print(ds['train']['text'])
    # print(summarizer.summarize(ds['train']['text']))
    # print(summarizer.summarize(load_examples(args.input)))
    # output = json.dumps(pipeline(load_examples(args.input)))
    # print(output)
