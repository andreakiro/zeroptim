from argparse import ArgumentParser
from zeroptim.configs import load
from zeroptim.trainer import ZeroptimTrainer

# parse command line
parser = ArgumentParser()
parser.add_argument("--config", type=str, help="path to config file", required=True)
parser.add_argument("--epochs", type=int, help="number of training epochs", default=10)
parser.add_argument("--max_iters", type=int, help="max total iterations", default=None)
parser.add_argument("--landscape", type=str, default="partial", choices=["batch", "partial", "global"])
parser.add_argument("--directed", type=bool, default=False)
args = parser.parse_args()

# init trainer
config = load(args.config)
trainer = ZeroptimTrainer.from_config(config)

trainer.DIRECTED = args.directed
trainer.LANDSCAPE = args.landscape

trainer.train(epochs=args.epochs, max_iters=args.max_iters)
