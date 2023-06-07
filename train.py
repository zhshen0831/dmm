import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from models import EncoderDecoder
from data_utils import DataLoader
import time, os, shutil, logging, h5py


torch.backends.cudnn.enabled = False
device = torch.device("cuda:1")

def NLLcriterion(vocab_size):
    weight = torch.ones(vocab_size)
    weight[0] = 0
    criterion = nn.NLLLoss(weight, size_average=False)
    return criterion

def batchloss(output, target, generator, lossF, g_batch):
    batch = output.size(1)
    loss = 0
    target = target[1:]
    for o, t in zip(output.split(g_batch),
                    target.split(g_batch)):
        o = o.view(-1, o.size(2))
        o = generator(o)
        t = t.view(-1)
        loss += lossF(o, t)

    return loss.div(batch)


def init_parameters(model):
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)

def savecheckpoint(state, is_best, filename="checkpoint.pt"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best_model.pt')

def validate(valData, model, lossF, args):
    m0, m1 = model
    m0.eval()
    m1.eval()

    num_iteration = valData.size // args.batch
    if valData.size % args.batch > 0: num_iteration += 1

    total_loss = 0
    for iteration in range(num_iteration):
        input, lengths, target = valData.getbatch()
        with torch.no_grad():
            input = Variable(input)
            lengths = Variable(lengths)
            target = Variable(target)
        if args.cuda and torch.cuda.is_available():
            input, lengths, target = input.to(device), lengths.to(device), target.to(device)
        output = m0(input, lengths, target)
        loss = batchloss(output, target, m1, lossF, 2)
        total_loss += loss * output.size(1)
    m0.train()
    m1.train()
    return total_loss.item() / valData.size

def train(args):
    logging.basicConfig(filename="training.log", level=logging.INFO)

    trainsrc = os.path.join(args.data, "train.src")
    traintrg = os.path.join(args.data, "train.trg")
    trainData = DataLoader(trainsrc, traintrg, args.batch, args.bucketsize)
    print("Read training data")
    trainData.load(args.max_num_line)

    valsrc = os.path.join(args.data, "val.src")
    valtrg = os.path.join(args.data, "val.trg")
    if os.path.isfile(valsrc) and os.path.isfile(valtrg):
        valData = DataLoader(valsrc, valtrg, args.batch, args.bucketsize, True)
        print("Read validation data")
        valData.load()
        print("Load validation data")
    else:
        print("No validation data")

    if args.criterion_name == "NLL":
        criterion = NLLcriterion(args.input_cell_size)
        lossF = lambda o, t: criterion(o, t)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
        lossF = lambda o, t: criterion(o, t)

    m0 = EncoderDecoder(args.input_cell_size,
                        args.output_road_size,
                        args.embedding_size,
                        args.hidden_size,
                        args.num_layers,
                        args.dropout,
                        args.bidirectional)
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.output_road_size),
                       nn.LogSoftmax())
    print(args.cuda)
    if args.cuda and torch.cuda.is_available():
        print("=> training with GPU")
        m0.to(device)
        m1.to(device)
        criterion.to(device)
    else:
        print("=> training with CPU")

    m0_optimizer = torch.optim.Adam(m0.parameters(), lr=args.learning_rate)
    m1_optimizer = torch.optim.Adam(m1.parameters(), lr=args.learning_rate)

    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        logging.info("Restore training @ {}".format(time.ctime()))
        checkpoint = torch.load(args.checkpoint)
        args.start_iteration = checkpoint["iteration"]
        best_prec_loss = checkpoint["best_prec_loss"]
        m0.load_state_dict(checkpoint["m0"])
        m1.load_state_dict(checkpoint["m1"])
        m0_optimizer.load_state_dict(checkpoint["m0_optimizer"])
        m1_optimizer.load_state_dict(checkpoint["m1_optimizer"])
    else:
        print("no checkpoint".format(args.checkpoint))
        logging.info("Start training".format(time.ctime()))
        best_prec_loss = float('inf')

    num_iteration = args.epochs * sum(trainData.allocation) // args.batch
    print("Start at {} "
          "and end at {}".format(args.start_iteration, num_iteration-1))
    for iteration in range(args.start_iteration, num_iteration):
        try:
            input, lengths, target = trainData.getbatch()
            input, lengths, target = Variable(input), Variable(lengths), Variable(target)
            if args.cuda and torch.cuda.is_available():
                input, lengths, target = input.to(device), lengths.to(device), target.to(device)

            m0_optimizer.zero_grad()
            m1_optimizer.zero_grad()
            output = m0(input, lengths, target)
            loss = batchloss(output, target, m1, lossF, args.g_batch)
            loss.backward()
            clip_grad_norm(m0.parameters(), 5)
            clip_grad_norm(m1.parameters(), 5)
            m0_optimizer.step()
            m1_optimizer.step()
            avg_loss = loss.item() / target.size(0)
            if iteration % args.print == 0:
                print("Iteration: {}\tLoss: {}".format(iteration, avg_loss))
            if iteration % args.save == 0 and iteration > 0:
                prec_loss = validate(valData, (m0, m1), lossF, args)
                if prec_loss < best_prec_loss:
                    best_prec_loss = prec_loss
                    logging.info("Best model with loss {} at iteration {} @ {}"\
                                 .format(best_prec_loss, iteration, time.ctime()))
                    is_best = True
                else:
                    is_best = False

                print("Save the model at iteration {} validation loss {}"\
                      .format(iteration, prec_loss))

                savecheckpoint({
                    "iteration": iteration,
                    "best_prec_loss": best_prec_loss,
                    "m0": m0.state_dict(),
                    "m1": m1.state_dict(),
                    "m0_optimizer": m0_optimizer.state_dict(),
                    "m1_optimizer": m1_optimizer.state_dict()
                }, is_best)
        except KeyboardInterrupt:
            break
