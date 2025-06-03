import torch
from torch.nn.utils.rnn import pad_sequence

def gather_csr(src, indptr):
    return [src[indptr[i]:indptr[i+1]] for i in range(len(indptr) - 1)]

def gather_csr_padded(src, indptr, padding_value=0):
    segments = [src[indptr[i]:indptr[i+1]] for i in range(len(indptr) - 1)]
    return pad_sequence(segments, batch_first=True, padding_value=padding_value)

def segment_csr(src, indptr, reduce="sum"):
    out = []
    for i in range(len(indptr) - 1):
        segment = src[indptr[i]:indptr[i+1]]
        if reduce == "sum":
            out.append(segment.sum(dim=0))
        elif reduce == "mean":
            out.append(segment.mean(dim=0))
        elif reduce == "max":
            out.append(segment.max(dim=0).values)
        else:
            raise ValueError(f"Unsupported reduce type: {reduce}")
    return torch.stack(out, dim=0)