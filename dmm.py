import argparse
from train import train
from evaluate import evaluator, dmm
from evaluate_bs import evaluate_bs

parser = argparse.ArgumentParser(description="train.py")

parser.add_argument("-data", default="data",
    help="Path to train and test data")

parser.add_argument("-checkpoint", default="checkpoint.pt",
    help="Saved checkpoint")

parser.add_argument("-pretrained_embedding", default=None,
    help="Path to the pretrained cell embedding")

parser.add_argument("-num_layers", type=int, default=3,
    help="GRU layers")

parser.add_argument("-hidden_size", type=int, default=128,
    help="Hidden state size in GRUs")

parser.add_argument("-embedding_size", type=int, default=128,
    help="Cell embedding size")

parser.add_argument("-print", type=int, default=50,
    help="Print for x iters")


parser.add_argument("-dropout", type=float, default=0.1,
    help="Dropout")

parser.add_argument("-learning_rate", type=float, default=0.001)

parser.add_argument("-g_batch", type=int, default=2,
    help="The maximum number of cells to generate each time")

parser.add_argument("-start_iteration", type=int, default=0)

parser.add_argument("-epochs", type=int, default=15,
    help="The number of training epochs")

parser.add_argument("-save", type=int, default=1000,
    help="Save frequency")

parser.add_argument("-cuda", type=bool, default=True,
    help="True for GPU")

parser.add_argument("-criterion_name", default="CE")

parser.add_argument("-max_num_line", type=int, default=1000000)

parser.add_argument("-bidirectional", type=bool, default=False,
    help="True for bidirectional rnn in encoder")

parser.add_argument("-max_length", default=2000,
    help="The maximum length of the target sequence")

parser.add_argument("-mode", type=int, default=0)

parser.add_argument("-batch", type=int, default=128,
    help="The batch size")

parser.add_argument("-bucketsize", default=[(8,80),(12,120),(16,160),(20,200),(24,240),(28,280),(32,320)],
    help="Bucket size for training")

parser.add_argument("-input_cell_size", type=int, default=0,
    help="Cell size")

parser.add_argument("-output_road_size", type=int, default=52989,
    help="Road size")

args = parser.parse_args()

if args.mode == 1:
    evaluator(args)
elif args.mode == 2:
    evaluate_bs(args)
else:
    train(args)
