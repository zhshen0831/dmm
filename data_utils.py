import numpy as np
import torch
from torch.autograd import Variable
from funcy import merge

def argsort(seq):
    return [x for x,y in sorted(enumerate(seq),
                                key = lambda x: len(x[1]),
                                reverse=True)]

def pad_array(a, max_length, PAD=0):
    return np.concatenate((a, [PAD]*(max_length - len(a))))

def pad_arrays(a):
    max_length = max(map(len, a))
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a).astype(np.int)
    return torch.LongTensor(a)

def pad_arrays_pair(src, trg):
    assert len(src) == len(trg), "source and target should have the same length"
    idx = argsort(src)
    src = list(np.array(src)[idx])
    trg = list(np.array(trg)[idx])
    lengths = list(map(len, src))
    lengths = torch.LongTensor(lengths)
    src = pad_arrays(src)
    trg = pad_arrays(trg)
    return src.t().contiguous(), lengths.view(1, -1), trg.t().contiguous()

def invpermute(p):
    p = np.asarray(p)
    invp = np.empty_like(p)
    for i in range(p.size):
        invp[p[i]] = i
    return invp

def pad_arrays_keep_invp(src):
    idx = argsort(src)
    src = list(np.array(src)[idx])
    lengths = list(map(len, src))
    lengths = torch.LongTensor(lengths)
    src = pad_arrays(src)
    invp = torch.LongTensor(invpermute(idx))
    return src.t().contiguous(), lengths.view(1, -1), invp


class DataLoader():
    def __init__(self, srcfile, trgfile, batch, bucketsize, validate=False):
        self.srcfile = srcfile
        self.trgfile = trgfile
        self.batch = batch
        self.validate = validate
        self.bucketsize = bucketsize

    def insert(self, s, t):
        for i in range(len(self.bucketsize)):
            if len(s) <= self.bucketsize[i][0] and len(t) <= self.bucketsize[i][1]:
                self.srcdata[i].append(np.array(s, dtype=np.int32))
                self.trgdata[i].append(np.array(t, dtype=np.int32))
                return 1
        return 0

    def load(self, max_num_line=0):
        self.srcdata = [[] for _ in range(len(self.bucketsize))]
        self.trgdata = [[] for _ in range(len(self.bucketsize))]
        srcstream, trgstream = open(self.srcfile, 'r'), open(self.trgfile, 'r')
        num_line = 0
        for (s, t) in zip(srcstream, trgstream):
            s = [int(x) for x in s.split()]
            t = [1] + [int(x) for x in t.split()] + [2]
            num_line += self.insert(s, t)
            if num_line >= max_num_line and max_num_line > 0: break
            if num_line % 100000 == 0:
                print("Read line {}".format(num_line))

        if self.validate == True:
            self.srcdata = np.array(merge(*self.srcdata))
            self.trgdata = np.array(merge(*self.trgdata))
            self.start = 0
            self.size = len(self.srcdata)
        else:
            self.srcdata = list(map(np.array, self.srcdata))
            self.trgdata = list(map(np.array, self.trgdata))
            self.allocation = list(map(len, self.srcdata))
            self.p = np.array(self.allocation) / sum(self.allocation)
        srcstream.close(), trgstream.close()

    def getbatch(self):
        if self.validate == True:
            src = self.srcdata[self.start:self.start+self.batch]
            trg = self.trgdata[self.start:self.start+self.batch]
            self.start += self.batch
            if self.start >= self.size:
                self.start = 0
            return pad_arrays_pair(src, trg)
        else:
            sample = np.random.multinomial(1, self.p)
            bucket = np.nonzero(sample)[0][0]
            idx = np.random.choice(len(self.srcdata[bucket]), self.batch)
            return pad_arrays_pair(self.srcdata[bucket][idx],
                                   self.trgdata[bucket][idx])


class DataOrderScaner():
    def __init__(self, srcfile, batch):
        self.srcfile = srcfile
        self.batch = batch
        self.srcdata = []
        self.start = 0
    def load(self, max_num_line=0):
        num_line = 0
        with open(self.srcfile, 'r') as srcstream:
            for s in srcstream:
                s = [int(x) for x in s.split()]
                self.srcdata.append(np.array(s, dtype=np.int32))
                num_line += 1
                if max_num_line > 0 and num_line >= max_num_line:
                    break
        self.size = len(self.srcdata)
        self.start = 0
    def getbatch(self):
        if self.start >= self.size:
            return None, None, None
        src = self.srcdata[self.start:self.start+self.batch]
        self.start += self.batch
        return pad_arrays_keep_invp(src)
