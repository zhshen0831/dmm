import torch
import torch.nn as nn
from torch.autograd import Variable
from models import EncoderDecoder
from data_utils import DataOrderScaner
import os, h5py

from math import log


def evaluate1(src, model, max_length):
    m0, m1 = model
    K = 10
    length = len(src)
    first = str(src[0])
    src = Variable(torch.LongTensor(src))
    src = src.view(-1, 1)
    length = Variable(torch.LongTensor([[length]]))

    encoder_hn, H = m0.encoder(src, length)
    h = m0.encoder_hn2decoder_h0(encoder_hn)

    start = 1

    sequences = [[[start], [h], 1.0]]
    result = []

    for _ in range(max_length):
        all_candidates = []
        for i in range(len(sequences)):
            seq, h, score = sequences[i]
            input = Variable(torch.LongTensor([[seq[-1]]]))
            h2 = h[-1]
            o, h1 = m0.decoder(input, h2, H)
            o = o.view(-1, o.size(2))
            o = m1(o)
            prob, word_id = o.data.topk(K)
            pro = prob.tolist()[0]
            word = word_id.tolist()[0]

            for j in zip(word, pro):
                if j[0] == 2:
                    result.append(seq[1:])
                    candidate = [seq, h, score]
                    all_candidates.append(candidate)
                else:
                    candidate = [seq + [j[0]], h + [h1], (score * j[1])/(1+len(seq))]
                    all_candidates.append(candidate)
        if len(result) >= K:
            break
        ordered = sorted(all_candidates, key=lambda tup: tup[2])
        sequences = ordered[:K]
    return result

def evaluate_bs(args):
    file1 = open(args.data + 'val.src', encoding='utf8',
                 errors='ignore')
    linessrc = []
    for line in file1:
        linessrc.append(line.strip('\n').strip())

    linestrg = []
    file2 = open(args.data + 'val.trg', encoding='utf8',
                 errors='ignore')
    for line in file2:
        linestrg.append(line.strip('\n').strip())
    m0 = EncoderDecoder(args.input_cell_size, args.output_road_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.output_road_size),
                       nn.LogSoftmax())
    if os.path.isfile(args.checkpoint):
        print("loading checkpoint")
        checkpoint = torch.load(args.checkpoint)
        m0.load_state_dict(checkpoint["m0"])
        m1.load_state_dict(checkpoint["m1"])
        i=0
        while i<1:
            try:
                p = 0
                r = 0
                count = 0
                for line in linessrc:

                    line1 = [int(x) for x in line.split(' ')]
                    trglist = evaluate1(line1, (m0, m1), args.max_length)

                    maxp = 0
                    maxr = 0
                    ltrg = linestrg[count]
                    ttrg = [int(x) for x in ltrg.strip().split(' ')]
                    if len(trglist) > 0:
                        for trg in trglist:
                            if len(trg) > 0:
                                intersact = set(ttrg) & set(trg)
                                p1 = len(intersact) / len(trg)
                                r1 = len(intersact) / len(ttrg)
                                if p1 > maxp:
                                    maxp = p1
                                if r1 > maxr:
                                    maxr = r1
                    p = p + maxp
                    r = r + maxr
                    count = count + 1
                    print("p {} r {}".format(maxp, maxr))
                print("p {} r {}".format(p/len(linessrc), r/len(linessrc)))
                i=i+1
            except KeyboardInterrupt:
                break
    else:
        print("no checkpoint")
