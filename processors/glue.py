import csv
import os
import sys
import json


PROMPT = {
    "MNLI": " ? [MASK] ! ",
    "QQP": " ? [MASK] ! ",
    "QNLI": " ? [MASK] ! ",
    "SST-2": " In summary , my review is [MASK] . ",
    "CoLA": " The grammar of the following sentence is [MASK] , ",
    "STS-B": " ? [MASK] !! ",
    "MRPC": " ? [MASK] , ",
    "RTE": " ? [MASK] !! ",
    "FOLIO": " ? [MASK] ! ",
}

def prompt_text(task, text_a, text_b):
    text = "[CLS] "
    if task == "MNLI":
        text += text_a
        text += " [SEP] "
        text += PROMPT[task]
        text += text_b
    elif task == "QQP":
        text += text_a
        text += " [SEP] "
        text += PROMPT[task]
        text += text_b
    elif task == "QNLI":
        text += text_a
        text += " [SEP] "
        text += PROMPT[task]
        text += text_b
    elif task == "SST-2":
        text += text_a
        text += " [SEP] "
        text += PROMPT[task]
    elif task == "CoLA":
        text += PROMPT[task]
        text += " [SEP] "
        text += text_a
    elif task == "STS-B":
        text += text_a
        text += " [SEP] "
        text += PROMPT[task]
        text += text_b
    elif task == "MRPC":
        text += text_a
        text += " [SEP] "
        text += PROMPT[task]
        text += text_b
    elif task == "RTE":
        text += text_a
        text += " [SEP] "
        text += PROMPT[task]
        text += text_b
    elif task == "FOLIO":
        text += text_a
        text += " [SEP] "
        text += PROMPT[task]
        text += text_b
    text += " [SEP]"
    return text


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


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")),
            "train",
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev",
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_label_words(self):
        return ["No", "Yes"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                    prompted=prompt_text("MRPC", text_a, text_b)
                )
            )
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")),
            "train",
        )

    def get_dev_examples(self, data_dir, dev_type="dev_matched"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, dev_type+".tsv")),
            dev_type,
        )

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def get_label_words(self):
        return ["No", "Yes", "Maybe"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                    prompted=prompt_text("MNLI", text_a, text_b)
                )
            )
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")),
            "train",
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev",
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_label_words(self):
        return ["incorrect", "correct"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=None,
                    label=label,
                    prompted=prompt_text("CoLA", text_a, None)
                )
            )
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")),
            "train",
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev",
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_label_words(self):
        return ["terrible", "great"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=None,
                    label=label,
                    prompted=prompt_text("SST-2", text_a, None)
                )
            )
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")),
            "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev"
        )

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test"
        )

    def get_labels(self):
        """See base class."""
        return [None]

    def get_label_words(self):
        return ["No", "Yes"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = f"{set_type}-{line[0]}"
            text_a = line[7]
            text_b = line[8]
            label = None if set_type == "test" else line[-1]
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                    prompted=prompt_text("STS-B", text_a, text_b)
                )
            )
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")),
            "train",
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev",
        )

    def get_labels(self):
        """See base class."""
        return ["not_entailment", "entailment"]

    def get_label_words(self):
        return ["No", "Yes"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                    prompted=prompt_text("QNLI", text_a, text_b)
                )
            )
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")),
            "train",
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev",
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_label_words(self):
        return ["No", "Yes"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) < 6:
                print("warning! empty sample: {}".format(line))
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[5]
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                    prompted=prompt_text("QQP", text_a, text_b)
                )
            )
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")),
            "train",
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev",
        )

    def get_labels(self):
        """See base class."""
        return ["not_entailment", "entailment"]

    def get_label_words(self):
        return ["No", "Yes"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label,
                    prompted=prompt_text("RTE", text_a, text_b)
                )
            )
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")),
            "train",
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev",
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = line[2]
            label = line[3]
            examples.append(
                InputExample(guid=guid,
                             text_a=text_a,
                             text_b=text_b,
                             label=label))
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

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
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

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


PROCESSORS = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qnli": QnliProcessor,
    "qqp": QqpProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
}
