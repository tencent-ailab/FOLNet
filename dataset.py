import torch
import json
import math
import h5py
import numpy as np
import random
from torch.utils.data import (
    IterableDataset,
    Dataset,
    DataLoader,
    RandomSampler
)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break
    if len(tokens_a) > len(tokens_b):
        del tokens_a[0]
    else:
        tokens_b.pop()


class Hdf5Dataset(Dataset):
    def __init__(self, input_file, args):
        self.input_file = input_file
        f = h5py.File(input_file, "r")
        shard_name = input_file.split('/')[-1]
        print("h5py instance created for shard: {}".format(shard_name))
        keys = [
            'input_ids',
            'input_mask',
            'segment_ids',
            'masked_lm_positions',
            'masked_lm_ids',
            'next_sentence_labels'
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        print("data loaded into memory for shard: {}".format(shard_name))
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        [
            input_ids, input_mask, segment_ids, masked_lm_positions,
            masked_lm_ids, next_sentence_labels
        ] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5
            else torch.from_numpy(np.asarray(input[index].astype(np.int64)))
            for indice, input in enumerate(self.inputs)
        ]
        masked_lm_labels = torch.full(masked_lm_ids.shape, -1, dtype=torch.long)
        index = masked_lm_ids.shape[0]
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[:index] = masked_lm_ids[:index]

        return {
            "token_input": input_ids,
            "seg_input": segment_ids,
            "mask_input": input_mask,
            "masked_lm_ids": masked_lm_ids,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_labels": masked_lm_labels,
            "is_random_next": next_sentence_labels,
        }


class BinDataset(IterableDataset):
    def __init__(self, input_file, args):
        super().__init__()
        self.pretrain_loss = args.pretrain_loss
        self.corpus_path = input_file
        self.meta_info = self._get_data_meta_info()
        self.header = self._get_data_header()
        self.field_boundary = self._get_field_boundary()
        self.field_dtype = self._get_field_dtype()
        self.bytes_per_sample = self.meta_info["bytes_per_sample"]
        self.num_samples = self.meta_info["num_samples"]
        self.start = 0
        self.end = self.num_samples - 1
        shard_name = input_file.split('/')[-1]
        print("loading binary shard: {}".format(shard_name))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_start, worker_samples = self._set_file_range(worker_info)
        with open(self.corpus_path, 'rb') as data_file:
            data_file.seek(worker_start)
            for _ in range(worker_samples):
                sample = data_file.read(self.bytes_per_sample)
                output = self._parse_bytes(sample)
                output = {
                    "tok_seq": output["token_input"],
                    "token_type_ids": output["seg_input"],
                    "mlm_positions": output["masked_lm_positions"],
                    "mlm_labels": output["masked_lm_labels"],
                    "nsp_label": output["is_random_next"],
                }
                yield {
                    key: torch.tensor(value, dtype=torch.long)
                    for key, value in output.items()
                }

    def __len__(self):
        return self.num_samples

    def _set_file_range(self, worker_info):
        if worker_info is None:
            worker_start = self.start
            worker_samples = self.num_samples
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(math.ceil(self.num_samples/float(num_workers)))
            worker_start_line = self.start + worker_id * per_worker
            worker_start = worker_start_line * self.bytes_per_sample
            worker_samples = min(per_worker, self.end - worker_start_line + 1)
        return worker_start, worker_samples

    def _get_data_header(self):
        input_header_path = self.corpus_path[:-4] + ".header"
        with open(input_header_path, 'r') as f_header:
            header = f_header.readline().strip().split('\t')
        return header

    def _get_data_meta_info(self):
        with open(self.corpus_path[:-4]+".met") as file_meta:
            meta_info = json.load(file_meta)
        return meta_info

    def _get_field_boundary(self):
        meta = self.meta_info
        field_s = {k: meta["offset_"+k] for k in self.header}
        field_e = {k: meta["offset_"+k]+meta["bytes_"+k] for k in self.header}
        return (field_s, field_e)

    def _get_field_dtype(self):
        field_dtype = {k: self.meta_info["dtype_"+k] for k in self.header}
        return field_dtype

    def _parse_bytes(self, sample):
        output = {
            k: np.frombuffer(
                sample[self.field_boundary[0][k]:self.field_boundary[1][k]],
                dtype=self.field_dtype[k]
            )
            for k in self.header
        }
        return output


class DynDataset(IterableDataset):
    def __init__(self, input_file, args):
        super().__init__()
        self.pretrain_loss = args.pretrain_loss
        self.corpus_path = input_file
        self.masked_lm_prob = args.masked_lm_prob
        self.max_predictions_per_seq = args.max_predictions_per_seq
        self.vocab_size = args.vocab_size
        self.pad_token = args.pad_token
        self.cls_token = args.cls_token
        self.sep_token = args.sep_token
        self.mask_token = args.mask_token
        self.meta_info = self._get_data_meta_info()
        self.header = self._get_data_header()
        self.field_boundary = self._get_field_boundary()
        self.field_dtype = self._get_field_dtype()
        self.bytes_per_sample = self.meta_info["bytes_per_sample"]
        self.num_samples = self.meta_info["num_samples"]
        self.start = 0
        self.end = self.num_samples - 1
        shard_name = input_file.split('/')[-1]
        print("loading binary shard: {}".format(shard_name))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_start, worker_samples = self._set_file_range(worker_info)
        with open(self.corpus_path, 'rb') as data_file:
            data_file.seek(worker_start)
            for _ in range(worker_samples):
                sample = data_file.read(self.bytes_per_sample)
                sample = self._parse_bytes(sample)
                orig_tokens = sample["token_input"].tolist()
                tokens, seg_ids, nsp, _ = self._create_nsp_predictions(
                    orig_tokens,
                    self.cls_token,
                    self.sep_token,
                    self.pad_token
                )
                ids, positions, labels = self._create_masked_lm_predictions(
                    tokens,
                    self.masked_lm_prob,
                    self.max_predictions_per_seq,
                    self.vocab_size,
                    self.cls_token,
                    self.sep_token,
                    self.pad_token,
                    self.mask_token
                )
                pad_len = self.max_predictions_per_seq - len(positions)
                positions.extend([self.pad_token]*pad_len)
                labels.extend([-1]*pad_len)
                nsp = [nsp]
                if "SCL" in self.pretrain_loss:
                    tokens_, seg_ids_, nsp_, _ = self._create_nsp_predictions(
                        orig_tokens,
                        self.cls_token,
                        self.sep_token,
                        self.pad_token
                    )
                    ids_, pos_, labels_ = self._create_masked_lm_predictions(
                        tokens_,
                        self.masked_lm_prob,
                        self.max_predictions_per_seq,
                        self.vocab_size,
                        self.cls_token,
                        self.sep_token,
                        self.pad_token,
                        self.mask_token
                    )
                    pad_len = self.max_predictions_per_seq - len(pos_)
                    pos_.extend([self.pad_token]*pad_len)
                    labels_.extend([-1]*pad_len)
                    nsp_ = [nsp_]
                    ids = [ids, ids_]
                    seg_ids = [seg_ids, seg_ids_]
                    positions = [positions, pos_]
                    labels = [labels, labels_]
                    nsp = [nsp, nsp_]
                output = {
                    "tok_seq": ids,
                    "token_type_ids": seg_ids,
                    "mlm_positions": positions,
                    "mlm_labels": labels,
                    "nsp_label": nsp,
                }
                yield {
                    k: torch.tensor(v, dtype=torch.long)
                    for k, v in output.items()
                }

    def __len__(self):
        return self.num_samples

    def _set_file_range(self, worker_info):
        if worker_info is None:
            worker_start = self.start
            worker_samples = self.num_samples
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = int(math.ceil(self.num_samples/float(num_workers)))
            worker_start_line = self.start + worker_id * per_worker
            worker_start = worker_start_line * self.bytes_per_sample
            worker_samples = min(per_worker, self.end - worker_start_line + 1)
        return worker_start, worker_samples

    def _get_data_header(self):
        input_header_path = self.corpus_path[:-4] + ".header"
        with open(input_header_path, 'r') as f_header:
            header = f_header.readline().strip().split('\t')
        return header

    def _get_data_meta_info(self):
        with open(self.corpus_path[:-4]+".met") as file_meta:
            meta_info = json.load(file_meta)
        return meta_info

    def _get_field_boundary(self):
        meta = self.meta_info
        field_s = {k: meta["offset_"+k] for k in self.header}
        field_e = {k: meta["offset_"+k]+meta["bytes_"+k] for k in self.header}
        return (field_s, field_e)

    def _get_field_dtype(self):
        field_dtype = {k: self.meta_info["dtype_"+k] for k in self.header}
        return field_dtype

    def _parse_bytes(self, sample):
        output = {
            k: np.frombuffer(
                sample[self.field_boundary[0][k]:self.field_boundary[1][k]],
                dtype=self.field_dtype[k]
            )
            for k in self.header
        }
        return output

    def _create_masked_lm_predictions(
        self,
        tokens,
        masked_lm_prob,
        max_predictions_per_seq,
        vocab_size,
        cls_token,
        sep_token,
        pad_token,
        mask_token
    ):
        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == cls_token or token == sep_token or token == pad_token:
                continue
            cand_indexes.append(i)
        random.shuffle(cand_indexes)
        output_tokens = list(tokens)
        num_to_predict = min(
            max_predictions_per_seq,
            max(1, int(round(len(cand_indexes) * masked_lm_prob)))
        )
        masked_lms = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)
            masked_token = None
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = mask_token
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = random.randint(1, vocab_size-1)
                    while masked_token in (cls_token, sep_token, mask_token):
                        masked_token = random.randint(1, vocab_size-1)
            output_tokens[index] = masked_token
            masked_lms.append((index, tokens[index]))
        masked_lms = sorted(masked_lms, key=lambda x: x[0])
        masked_lm_positions = []
        masked_lm_labels = []
        for p in masked_lms:
            masked_lm_positions.append(p[0])
            masked_lm_labels.append(p[1])
        return (output_tokens, masked_lm_positions, masked_lm_labels)

    def _create_nsp_predictions(
        self,
        tokens,
        cls_token,
        sep_token,
        pad_token
    ):
        max_seq_length = len(tokens)
        pad_idx = None
        for n in range(len(tokens)-1, -1, -1):
            if tokens[n] != pad_token:
                pad_idx = n + 1
                break
        if pad_idx:
            tokens = tokens[:pad_idx]
        assert tokens[0] == cls_token
        assert tokens[-1] == sep_token
        tokens = tokens[1:-1]

        sep_idx = int(len(tokens) / 2) - 1
        shift = np.random.geometric(p=0.2)
        if random.random() < 0.5:
            shift *= -1
        sep_idx += shift
        sep_idx = max(sep_idx, 0)
        sep_idx = min(sep_idx, len(tokens))

        tokens_a = tokens[:sep_idx]
        tokens_b = tokens[sep_idx:]
        truncate_seq_pair(tokens_a, tokens_b, max_seq_length-3)
        is_random_next = 0
        if random.random() < 0.5:
            is_random_next = 1
            tokens_a, tokens_b = tokens_b, tokens_a
        tokens = [cls_token] + tokens_a + [sep_token] + tokens_b + [sep_token]
        segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)
        sep_pos = [len(tokens_a)+1, len(tokens_a)+len(tokens_b)+2]
        assert len(tokens) <= max_seq_length
        while len(tokens) < max_seq_length:
            tokens.append(pad_token)
            segment_ids.append(0)
        assert len(tokens) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return tokens, segment_ids, is_random_next, sep_pos
