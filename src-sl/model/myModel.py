from os.path import join

import torch
from torch import nn
from transformers import AutoModel, AutoModelWithLMHead
from reader.tags import *


class BertNER(nn.Module):
    def __init__(self, args, cfg):
        super(BertNER, self).__init__()

        pretrained_cache = join(args.pretrained_cache_dir, cfg.encoder_model)
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, cache_dir=pretrained_cache)
        self.hidden_size = self.encoder.config.hidden_size
        self.emission_ffn = nn.Linear(self.hidden_size, len(ID2TAG))
        self.crossentropy = torch.nn.CrossEntropyLoss(ignore_index=TAG2ID['[END]'])

    def forward(self, encoder_inputs: dict, tag_ids: torch.Tensor):
        outputs = self.encoder(**encoder_inputs)

        encoded, _ = outputs

        emission = self.emission_ffn(encoded)
        emission = emission.permute(1, 2, 0)
        tag_ids = tag_ids.permute(1, 0)
        loss = self.crossentropy(emission, tag_ids)
        return loss

    def predict(self, inputs: dict):
        outputs = self.encoder(**inputs)
        encoded, _ = outputs
        emission = self.emission_ffn(encoded)
        pred_ids = torch.argmax(emission, -1)
        masks = inputs['attention_mask']

        tag_paths = []
        for i in range(emission.shape[0]):
            tag_paths.append([ID2TAG[idx] for idx in pred_ids[i][:sum(masks[i])]])
        return tag_paths
