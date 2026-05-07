import random
import cli
import os

from datasets import load_dataset
from baseline import TextRankSummarizer
from metrics import compute_rouge
from logger_config import setup_logger

_DATASET_NAME = "IlyaGusev/gazeta"

logger = setup_logger(__name__)

if __name__ == "__main__":
    args = cli.parse_args()

    summarizer = None
    match args.method:
        case cli.Method.TEXTRANK:
            summarizer = TextRankSummarizer()
        case cli.Method.NN:
            raise RuntimeError("Not implemented")

    if args.from_dataset > 0:
        refs: list[str] = []
        preds: list[str] = []

        os.environ["HF_HUB_OFFLINE"] = "1"
        ds = load_dataset(_DATASET_NAME)

        if (args.metrics):
            references  = []
            predictions = []

        for i in random.sample(range(0, len(ds['train'])), args.from_dataset):
            logger.info(f"Dataset instance no. {i+1}")
            logger.info(f"\n\ttext: {ds['train'][i]['text']}")
            logger.info(f"\n\treference summary: {ds['train'][i]['summary']}")

            summary = summarizer.summarize(ds['train'][i]['text'])
            logger.info(f"\n\tinferenced summary: {summary}")

            if args.metrics:
                references.append(ds['train'][i]['summary'])
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


    # summarizer = TextRankSummarizer()

    # ds = load_dataset("IlyaGusev/gazeta")
    # print(ds['train']['text'])
    # print(summarizer.summarize(ds['train']['text']))
    # print(summarizer.summarize(load_examples(args.input)))
    # output = json.dumps(pipeline(load_examples(args.input)))
    # print(output)
