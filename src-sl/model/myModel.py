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
        self.embeddings = self.encoder.get_input_embeddings()

    def _get_encoder_outputs(self, encoder_inputs):
        input_ids_1 = encoder_inputs['input_ids_1']
        input_ids_2 = encoder_inputs['input_ids_2']

        attention_mask = encoder_inputs['attention_mask']
        token_type_ids = encoder_inputs['token_type_ids']
        position_ids = encoder_inputs['position_ids']

        embed_1 = self.embeddings(input_ids_1)
        embed_2 = self.embeddings(input_ids_2)
        embed = (embed_1 + embed_2) / 2
        outputs = self.encoder(inputs_embeds=embed, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               position_ids=position_ids)
        return outputs

    def forward(self, encoder_inputs: dict, tag_ids: torch.Tensor):
        outputs = self._get_encoder_outputs(encoder_inputs)
        encoded, _ = outputs

        emission = self.emission_ffn(encoded)
        emission = emission.permute(1, 2, 0)
        tag_ids = tag_ids.permute(1, 0)
        loss = self.crossentropy(emission, tag_ids)
        return loss

    def predict(self, inputs: dict):
        outputs = self._get_encoder_outputs(inputs)
        encoded, _ = outputs
        emission = self.emission_ffn(encoded)
        pred_ids = torch.argmax(emission, -1)
        masks = inputs['loss_mask']

        tag_paths = []
        for i in range(emission.shape[0]):
            tag_paths.append([ID2TAG[idx] for idx in pred_ids[i][:sum(masks[i])]])
        return tag_paths
