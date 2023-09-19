import csv
import os
import sys
import json


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, prompted=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For
                    single sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second
                    sequence. Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                   specified for train and dev examples, but not for test
                   examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.prompted = prompted


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_id,
        mlm_positions=None,
        option_ids=None,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.mlm_positions = mlm_positions
        self.option_ids = option_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class FolioProcessor(DataProcessor):
    """Processor for the FOLIO data set."""

    def _read_jsonl(self, input_file):
        with open(input_file, 'r') as json_file:
            json_list = list(json_file)
        lines = []
        for json_str in json_list:
            lines.append(json.loads(json_str))
        return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "folio-train.jsonl")),
            "train",
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "folio-validation.jsonl")),
            "dev",
        )

    def get_labels(self):
        """See base class."""
        # return ["False", "True", "Unknown"] # dataset bug: "Unknown" in train
        return ["False", "True", "Uncertain"] # dataset bug: "Uncertain" in dev

    def get_label_words(self):
        return ["No", "Yes", "Maybe"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["premises"]
            text_a = [x.strip() for x in text_a]
            text_a = " ".join(text_a)
            text_b = line["conclusion"]
            label = line["label"]
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label
                )
            )
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    output_mode = "classification" if len(label_list) > 1 else "regression"
    label_map = {label: i for i, label in enumerate(label_list)}

    num_exceed = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        len_a = len(tokens_a)

        tokens_b = None
        len_b = 0
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            len_b = len(tokens_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        num_exceed += ((len_a + len_b) > max_seq_length - 3)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    perc_exceed = num_exceed/len(examples)
    print('#features', len(features))
    print("#perc-exceed-seq={:.2%}(>{})".format(perc_exceed, max_seq_length))
    return features, label_map


def convert_examples_to_zeroshot_features(
    examples,
    label_list,
    label_words,
    max_seq_length,
    tokenizer
):
    output_mode = "classification" if len(label_list) > 1 else "regression"
    label_map = {label: i for i, label in enumerate(label_list)}

    num_exceed = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        # tokenize prompted text and convert into ids
        tokens = tokenizer.tokenize(example.prompted)
        num_exceed += (len(tokens) > max_seq_length)
        orig_mask_position = tokens.index("[MASK]")
        tokens_a = tokens[1:orig_mask_position]
        tokens_b = tokens[orig_mask_position+1:-1]
        while len(tokens_a) + len(tokens_b) > max_seq_length - 3:
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop(0)
            else:
                tokens_b.pop(-1)
        tokens = ["[CLS]"] + tokens_a + ["[MASK]"] + tokens_b + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # generate other meta-info
        mlm_positions = [tokens.index("[MASK]")]
        input_mask = [1] * len(input_ids)
        option_ids = [tokenizer.tokenize(s)[0] for s in label_words]
        option_ids = tokenizer.convert_tokens_to_ids(option_ids)
        sep_position = tokens.index("[SEP]")
        segment_ids = [0]*(sep_position+1) + [1]*(len(tokens)-sep_position-1)
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # generate label
        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)
        # pack all features into outputs
        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                mlm_positions=mlm_positions,
                option_ids=option_ids
            )
        )
    perc_exceed = num_exceed/len(examples)
    print('#features', len(features))
    print("#perc-exceed-seq={:.2%}(>{})".format(perc_exceed, max_seq_length))
    return features, label_map


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


PROCESSORS = {
    "folio": FolioProcessor,
}
