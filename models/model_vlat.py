from functools import partial
from models.xbert import BertLMHeadModel
from transformers import ViTConfig, ViTModel,BeitConfig, BeitModel
from transformers import BertConfig, BertModel, BertTokenizer
import torch
import timm
import numpy as np
from torch import nn

import torch.nn.functional as F

torch.cuda.empty_cache()


# class ImageEncoder(nn.Module):
#     def __init__(self, pretrained=True):
#         super(ImageEncoder, self).__init__()
#         self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k') if pretrained else ViTModel(ViTConfig())

#     def forward(self, images):
#         outputs = self.vit_model(images)
#         image_embeddings = outputs.last_hidden_state
#         return image_embeddings


# class TextEncoder(nn.Module):
#     def __init__(self, model_name='bert-base-uncased'):
#         super(TextEncoder, self).__init__()
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)

#         text_config_encoder = BertConfig(vocab_size=30522, num_hidden_layers=12, num_attention_heads=12, hidden_size=768, layer_norm_eps=1e-12, max_position_embeddings=512, hidden_act="gelu", hidden_dropout_prob=0.1, initializer_range=0.02, intermediate_size=3072, attention_probs_dropout_prob=0.1)   
#         self.bert = BertModel(config=text_config_encoder)
#         # self.bert = BertModel.from_pretrained(model_name)

#     def forward(self, text):
#         outputs = self.bert(text['input_ids'], attention_mask=text['attention_mask'])
#         return outputs.last_hidden_state

def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))   


class SelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=8):
        super(SelfAttention, self).__init__()
        self.mhatt = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        attn_output, _ = self.mhatt(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x


class GuidedAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=8):
        super(GuidedAttention, self).__init__()
        self.mhatt = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x, y, x_mask=None, y_mask=None):
        # Cross-attention between x and y
        attn_output, _ = self.mhatt(x, y, y, attn_mask=x_mask)
        # print("attn_ouptut.shape",attn_output.shape)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size=768, num_heads=8, num_layers=6):
        super(EncoderDecoder, self).__init__()
        self.self_attention_layers = nn.ModuleList([SelfAttention(hidden_size, num_heads) for _ in range(num_layers)])
        self.guided_attention_layers = nn.ModuleList([GuidedAttention(hidden_size, num_heads) for _ in range(num_layers)])

    def forward(self, image_embeddings, text_embeddings, image_mask=None, text_mask=None):
        # Process image and text embeddings with self-attention
        for sa_layer in self.self_attention_layers:
            image_embeddings = sa_layer(image_embeddings, image_mask)
            text_embeddings = sa_layer(text_embeddings, text_mask)
        
        # Cross-attention (guided attention) between image and text
        for ga_layer in self.guided_attention_layers:
            image_embeddings = ga_layer(image_embeddings, text_embeddings, image_mask, text_mask)
            text_embeddings = ga_layer(text_embeddings, image_embeddings, text_mask, image_mask)
        
        return image_embeddings, text_embeddings


class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True, output_dim=768):
        super(ImageEncoder, self).__init__()
        self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k') if pretrained else ViTModel(ViTConfig(image_size=224, patch_size=16, embed_dim=768, mlp_ratio=4, qkv_bias=True, layer_norm_eps=1e-6))
    def forward(self, images):
        outputs = self.vit_model(images)
        # print(outputs)
        # print(outputs["last_hidden_state"].size()[:-1])
        return outputs.last_hidden_state[:, :-1, :]


class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', output_dim=768):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        text_config_encoder = BertConfig(vocab_size=30522, num_hidden_layers=12, num_attention_heads=12, hidden_size=768, layer_norm_eps=1e-12, max_position_embeddings=768, hidden_act="gelu", hidden_dropout_prob=0.1, initializer_range=0.02, intermediate_size=3072, attention_probs_dropout_prob=0.1)   
        self.bert = BertModel(config=text_config_encoder)
        # self.linear = nn.Linear(text_config_encoder.hidden_size, output_dim)

    def forward(self, text):
        outputs = self.bert(text['input_ids'], attention_mask=text['attention_mask'])
        # print(outputs)
        return outputs.last_hidden_state



class VLAT(nn.Module):
    def __init__(self):
        super(VLAT, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.encoder_decoder = EncoderDecoder()
        text_config_encoder = BertConfig(vocab_size=30522, num_hidden_layers=12, num_attention_heads=12, hidden_size=768, layer_norm_eps=1e-12, max_position_embeddings=512, hidden_act="gelu", hidden_dropout_prob=0.1, initializer_range=0.02, intermediate_size=3072, attention_probs_dropout_prob=0.1)   
        self.text_encoder_1 = BertModel(config=text_config_encoder)
        config_decoder = BertConfig.from_json_file("/home/beast/Desktop/Daniel/charan_anna/VLAT/configs/config_bert.json")
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6
        self.text_decoder = BertLMHeadModel.from_pretrained("bert-base-uncased", config=config_decoder)   
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 

    def make_mask(self, batch_size=16, num_heads=8, S=12):
        binary_mask = torch.randint(0, 2, (batch_size * num_heads, S, S), dtype=torch.bool)
        float_mask = binary_mask.float().masked_fill(binary_mask, float('-inf'))
        return float_mask

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True, reduction = "none")              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, reduction="none")                 
        answer_loss = output.logits
        answer_loss = answer_loss.view(input_ids.size(0),-1)
        
        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1,1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss],dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques,k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k,dim=1) 
        topk_ids = torch.gather(topk_ids, 1, rerank_id)    

        return topk_ids, topk_probs

    def forward(self, image, questions, answer, weights, k, train=True):
        
        image_features = self.image_encoder(image) # (16. 3, 224, 224) -->  (batch_size, num_patches+1, hidden_dim) # (16, 197, 768) # 16*1,51,296*3
        image_atts = torch.ones(image_features.size()[:-1],dtype=torch.long).to(image.device)
        # print("aaa",questions.attention_mask.shape, image_atts.shape)
        question_features = self.text_encoder(questions) #input_ids-->(16, sequence_length), (16, sequence_length,768) # (batch_size, sequence_length, hidden_dim), (16, 11, 768) #16*8448*3
        image_embeddings, text_embeddings = self.encoder_decoder(image_features, question_features) # need to Mask, with mask and without masking # mcan
        
        answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100) 
        
        # print("Encoder_decoder Block output", text_embeddings)
        if train:
            # question_output = self.text_encoder_1(questions.input_ids, 
            #                                         attention_mask = questions.attention_mask, 
            #                                         encoder_hidden_states = image_embeddings,
            #                                         encoder_attention_mask = image_atts,                             
            #                                         return_dict = True)
            question_states = []                
            question_atts = []
            # print(question_output.last_hidden_state.shape, text_embeddings.shape)  
            for b, n in enumerate(k):
                # question_states += [question_output["last_hidden_state"][b]]*n
                question_states += [text_embeddings[b]]*n
                question_atts += [questions.attention_mask[b]]*n 
            question_states = torch.stack(question_states,0)    
            question_atts = torch.stack(question_atts,0)   

            answer_output = self.text_decoder(answer.input_ids, 
                                                    attention_mask = answer.attention_mask, 
                                                    encoder_hidden_states = question_states,
                                                    encoder_attention_mask = question_atts,                  
                                                    labels = answer_targets,
                                                    return_dict = True, reduction = "none")
            
            loss = weights * answer_output.loss         
            loss = loss.sum()/image.size(0)

            # print("Loss:", loss)

            return loss
        else:
            # question_output = self.text_encoder_1(questions.input_ids, 
            #                                     attention_mask = questions.attention_mask, 
            #                                     encoder_hidden_states = image_embeddings,
            #                                     encoder_attention_mask = image_atts, return_dict = True)
            # print(question_output)                   
            topk_ids, topk_probs = self.rank_answer(text_embeddings, questions.attention_mask, 
                                                    answer.input_ids, answer.attention_mask, k) 
            return topk_ids, topk_probs
