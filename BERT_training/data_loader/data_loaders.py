from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm
import torch


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class IMDBDataloader(BaseDataLoader):
    def __init__(self, datapath, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, max_seq_len=64, adv_version=False):
        self.datapath = datapath
        self.training = training
        self.dataset = IMDBDataset(datapath, train=training, test=(not training), max_seq_len=max_seq_len, adv_version=adv_version)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.dataset.collate_fn)
 

class IMDBDataset(Dataset):
    def __init__(self, datapath, train=False, test=False, max_seq_len=64, adv_version=False):
        self.datapath = datapath
        self.data = []
        self.train = train
        self.test = test
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.loadData(datapath, train=self.train, test=self.test, adv_version=adv_version)
        

    def __len__(self):
        return len(self.data)

    def  __getitem__(self, index):
        return self.data[index]

    def loadData(self, datapath, train=False, test=False, adv_version=False):
        with open(datapath) as f:
            lines = f.readlines()
            lines = tqdm(lines, desc="Loading data")
            if adv_version:
                pass
            else: 
                for line in lines:
                    token = line[:-1][0], line[:-1][2:]
                    if train:
                        input_ids, attention_mask, token_type_ids = self.sentenceToIndex(token[1])
                        self.data.append({"label": int(token[0]), "text": token[1], "input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids})

    def sentenceToIndex(self, sentence, add_special_tokens=True):
        tokens = self.tokenizer.tokenize(sentence)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[:(self.max_seq_len - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_type_ids = [0] * len(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (self.max_seq_len - len(input_ids))
        input_ids += padding
        attention_mask += padding
        token_type_ids += padding

        assert len(input_ids) == self.max_seq_len
        assert len(attention_mask) == self.max_seq_len
        assert len(token_type_ids) == self.max_seq_len
        return input_ids, attention_mask, token_type_ids

    def collate_fn(self, datas):
        batch = {} 
        batch["text"] = [data["text"] for data in datas]
        batch["data"] = {}
        batch["data"]["input_ids"] = torch.tensor([data["input_ids"] for data in datas])
        batch["data"]["attention_mask"] = torch.tensor([data["attention_mask"] for data in datas])
        batch["data"]["token_type_ids"] = torch.tensor([data["token_type_ids"] for data in datas])
        if self.train: batch["data"]["labels"] = torch.tensor([data["label"] for data in datas])
        else: batch["data"]["labels"] = None
        return batch