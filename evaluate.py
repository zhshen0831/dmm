import torch
import torch.nn as nn
from torch.autograd import Variable
from models import EncoderDecoder
from data_utils import DataOrderScaner
import os, h5py
import time


def evaluate(src, model, max_length):
    m0, m1 = model
    length = len(src)
    src = Variable(torch.LongTensor(src))
    src = src.view(-1, 1)
    length = Variable(torch.LongTensor([[length]]))

    encoder_hn, H = m0.encoder(src, length)
    h = m0.encoder_hn2decoder_h0(encoder_hn)

    input = Variable(torch.LongTensor([[1]]))
    tg = []

    for _ in range(max_length):
        o, h = m0.decoder(input, h, H)
        o = o.view(-1, o.size(2))
        o = m1(o)
        _, id = o.data.topk(1)
        id = id[0][0]
        if id == 2:
            break
        tg.append(id.item())
        input = Variable(torch.LongTensor([[id]]))
    return tg

def evaluator(args):
    file1 = open(args.data+'val.src', encoding='utf8',
                 errors='ignore')
    linessrc = []
    for line in file1:
        linessrc.append(line.strip('\n').strip())

    linestrg = []
    file2 = open(args.data+'val.trg', encoding='utf8',
                 errors='ignore')
    for line in file2:
        linestrg.append(line.strip('\n').strip())

    m0 = EncoderDecoder(args.input_cell_size, args.output_road_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.output_road_size),
                       nn.LogSoftmax())
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        m0.load_state_dict(checkpoint["m0"])
        m1.load_state_dict(checkpoint["m1"])
        i=0
        while i<1:
            try:
                print("> ", end="")
                count=0
                p = 0
                r = 0
                t=0

                for line in linessrc:
                    line1 = [int(x) for x in line.split(' ')]
                    start = time.time()
                    trg = evaluate(line1, (m0, m1), args.max_length)
                    end = time.time()
                    t = t + end - start
                    if len(trg) > 0:
                        ttrg = [int(x) for x in linestrg[count].strip().split(' ')]
                        intersact = set(ttrg) & set(trg)
                        p = p + len(intersact) / len(trg)
                        r = r + len(intersact) / len(linestrg[count].split(' '))
                        print("p {} r {} count {}".format(len(intersact) / len(trg) , len(intersact) / len(linestrg[count].split(' ')), count))
                    count= count+1
                print("t {} ".format(t / count))
                print("p {} r {}".format(p / count, r / count))
                i=i+1
            except KeyboardInterrupt:
                break
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))

def dmm(args):
    m0 = EncoderDecoder(args.input_cell_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        m0.load_state_dict(checkpoint["m0"])
        if torch.cuda.is_available():
            m0.cuda()
        m0.eval()
        vecs = []
        scaner = DataOrderScaner(os.path.join(args.data, "trj.t"), args.batch)
        scaner.load()
        i = 0
        while True:
            if i % 10 == 0:
                print("{}: Encode {}".format(i, args.batch))
            i = i + 1
            src, lengths, invp = scaner.getbatch()
            if src is None: break
            src, lengths = Variable(src), Variable(lengths)
            if torch.cuda.is_available():
                src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
            h, _ = m0.encoder(src, lengths)
            h = m0.encoder_hn2decoder_h0(h)
            h = h.transpose(0, 1).contiguous()
            vecs.append(h[invp].cpu().data)

        vecs = torch.cat(vecs)
        vecs = vecs.transpose(0, 1).contiguous()
        path = os.path.join(args.data, "trj.h5")
        print("=> save vectors into {}".format(path))
        with h5py.File(path, "w") as f:
            for i in range(m0.num_layers):
                f["layer"+str(i+1)] = vecs[i].squeeze(0).numpy()
    else:
        print("=> no checkpoint found at the dir'{}'".format(args.checkpoint))
