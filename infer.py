import json
import random
import logging
import cli

from datasets import load_dataset
from baseline import TextRankSummarizer
from metrics import compute_rouge


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_examples(path: str) -> list[str]:
    texts: list[str] = []
    with open(path, 'r') as f_in:
        texts = json.loads(f_in.read())
    return texts


if __name__ == "__main__":
    args = cli.parse_args()

    ds = load_dataset("IlyaGusev/gazeta")

    summarizer = None
    match args.method:
        case cli.Method.TEXTRANK:
            summarizer = TextRankSummarizer()
        case cli.Method.NN:
            raise RuntimeError("Not implemented")

    if args.from_dataset > 0:
        refs: list[str] = []
        preds: list[str] = []

        for i in random.sample(range(0, len(ds['train'])), args.from_dataset):
            logger.info(f"Dataset instance no. {i+1}")
            logger.info(f"\n\ttext: {ds['train'][i]['text']}")
            logger.info(f"\n\treference summary: {ds['train'][i]['summary']}")

            pred_list = summarizer.summarize([ds['train'][i]['text']])
            pred = pred_list[0] if len(pred_list) > 0 else ""
            logger.info(f"\n\tinferenced summary: {pred}")

            if args.metrics:
                refs.append(ds['train'][i]['summary'])
                preds.append(pred)

        if args.metrics:
            # rouge = compute_rouge(preds, refs, limit=300)
            rouge = compute_rouge(refs, preds, limit=300)
            logger.info(
                "\nROUGE (mean over samples)\n"
                f"\trouge1: P={rouge['rouge1'].precision:.4f} R={rouge['rouge1'].recall:.4f} F1={rouge['rouge1'].f1:.4f}\n"
                f"\trouge2: P={rouge['rouge2'].precision:.4f} R={rouge['rouge2'].recall:.4f} F1={rouge['rouge2'].f1:.4f}\n"
                f"\trougeL: P={rouge['rougeL'].precision:.4f} R={rouge['rougeL'].recall:.4f} F1={rouge['rougeL'].f1:.4f}"
            )


    # summarizer = TextRankSummarizer()

    # ds = load_dataset("IlyaGusev/gazeta")
    # print(ds['train']['text'])
    # print(summarizer.summarize(ds['train']['text']))
    # print(summarizer.summarize(load_examples(args.input)))
    # output = json.dumps(pipeline(load_examples(args.input)))
    # print(output)
