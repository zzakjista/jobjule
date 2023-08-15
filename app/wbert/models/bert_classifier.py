import torch
import torch.nn as nn


class BERTClassifier(nn.Module):
    def __init__(self, bert, args):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.hidden_size = bert.config.hidden_size
        self.num_classes = args.num_classes
        self.dr_rate = args.dr_rate

        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        if self.dr_rate:
            self.dropout = nn.Dropout(p=self.dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        out = self.classifier(pooler)
        return out 