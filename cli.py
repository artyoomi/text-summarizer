from argparse import ArgumentParser, ArgumentTypeError
from enum import Enum, auto


# Method to generate summarization
class Method(Enum):
    TEXTRANK = auto()
    NN       = auto()

    @classmethod
    def from_string(cls, value: str):
        """Convert string to enum member"""
        try:
            # Try to match by name (case-sensitive)
            return cls[value]
        except Exception:
            # If not found, show error
            valid_names = [m.name for m in cls]
            raise ArgumentTypeError(
                f"Invalid value '{value}'. Choose from: {valid_names}"
            )

def parse_args():
    argparser = ArgumentParser(description="Inference script for autosummarization model")
    argparser.add_argument('-m', '--method', type=Method.from_string, default=Method.TEXTRANK)
    argparser.add_argument('-i', '--input', type=str, help="JSON file with list of sentences")
    argparser.add_argument(
        '-d',
        '--from-dataset',
        type=int,
        default=0,
        help="Get specified amount of sentences from dataset and make summarization for them"
    )
    argparser.add_argument(
        '--metrics',
        action='store_true',
        help="Enable ROUGE-1/2/L computation (for --from-dataset)"
    )
    return argparser.parse_args()
