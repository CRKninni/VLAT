from functools import partial

from transformers import BertConfig, BertLMHeadModel, BertForMaskedLM
from transformers import Swinv2Config, Swinv2Model
from transformers import RobertaConfig, RobertaModel
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.distill = False


        configuration = Swinv2Config(image_size=config["image_res"], patch_size=16, embed_dim=768, mlp_ratio=4, qkv_bias=True, layer_norm_eps=1e-6)
        self.visual_encoder = Swinv2Model(config=configuration)
        
        """
                {
        "architectures": [
            "BertForMaskedLM"
        ],
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "fusion_layer": 6,
        "encoder_width": 768
        }

        """

        text_config_encoder = RobertaConfig(vocab_size=30522, num_hidden_layers=12, num_attention_heads=12, hidden_size=768, layer_norm_eps=1e-12, max_position_embeddings=512, hidden_act="gelu", hidden_dropout_prob=0.1, initializer_range=0.02, intermediate_size=3072, attention_probs_dropout_prob=0.1)   
        self.text_encoder = RobertaModel(config=text_config_encoder, add_pooling_layer=False)
            
        config_decoder = BertConfig(vocab_size=30522, num_hidden_layers=12, num_attention_heads=12, hidden_size=768, layer_norm_eps=1e-12, max_position_embeddings=512, hidden_act="gelu", hidden_dropout_prob=0.1, initializer_range=0.02, intermediate_size=3072, attention_probs_dropout_prob=0.1)
        self.text_decoder = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)   

        if self.distill:
            self.visual_encoder_m =Swinv2Model(config=configuration)           
            self.text_encoder_m = RobertaModel(config=text_config_encoder, add_pooling_layer=False)  
            self.text_decoder_m = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)   
            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.text_decoder,self.text_decoder_m],
                               ]
            self.copy_params() 
            self.momentum = 0.995
        

    def forward(self, image, quesiton, answer=None, alpha=0, k=None, weights=None, train=True):
        
        image_embeds = self.visual_encoder(image)
        # print(image_embeds[0].size()) 
        image_atts = torch.ones(image_embeds[0].size()[:-1],dtype=torch.long).to(image.device)
        
        if train:               
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''          
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      
            # img_feat, lang_feat_mask, img_feat_mask
            # lang_feature, image_feature = self.mca(quesiton.input_ids, image_embeds, quesiton.attention_mask, image_atts)
            question_output = self.text_encoder(quesiton.input_ids, 
                                                attention_mask = quesiton.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = True)
            
            question_states = []                
            question_atts = []  
            for b, n in enumerate(k):
                question_states += [question_output.last_hidden_state[b]]*n
                question_atts += [quesiton.attention_mask[b]]*n 
            question_states = torch.stack(question_states,0)    
            question_atts = torch.stack(question_atts,0)     
            # print(question_atts.shape, question_states.shape)
            if self.distill:                    
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image) 
                    # lang_features, image maks, image features 
                    question_output_m = self.text_encoder_m(quesiton.input_ids, 
                                                            attention_mask = quesiton.attention_mask, 
                                                            encoder_hidden_states = image_embeds_m,
                                                            encoder_attention_mask = image_atts,                             
                                                            return_dict = True)    
                    question_states_m = []                
                    for b, n in enumerate(k):
                        question_states_m += [question_output_m.last_hidden_state[b]]*n
                    question_states_m = torch.stack(question_states_m,0)    

                    logits_m = self.text_decoder_m(answer.input_ids, 
                                                   attention_mask = answer.attention_mask, 
                                                   encoder_hidden_states = question_states_m,
                                                   encoder_attention_mask = question_atts,                                  
                                                   return_logits = True,
                                                  )                       

                answer_output = self.text_decoder(answer.input_ids, 
                                                  attention_mask = answer.attention_mask, 
                                                  encoder_hidden_states = question_states,
                                                  encoder_attention_mask = question_atts,                  
                                                  labels = answer_targets,
                                                  return_dict = True,   
                                                  soft_labels = F.softmax(logits_m,dim=-1),
                                                  alpha = alpha)   
            else:
                answer_output = self.text_decoder(answer.input_ids, 
                                                  attention_mask = answer.attention_mask, 
                                                  encoder_hidden_states = question_states,
                                                  encoder_attention_mask = question_atts,                  
                                                  labels = answer_targets,
                                                  return_dict = True)                      
            loss = weights * answer_output.loss         
            loss = loss.sum()/image.size(0)

            return loss
            

        else: 
            question_output = self.text_encoder(quesiton.input_ids, 
                                                attention_mask = quesiton.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts, return_dict = True)                    
            topk_ids, topk_probs = self.rank_answer(question_output.last_hidden_state, quesiton.attention_mask, 
                                                    answer.input_ids, answer.attention_mask, k) 
            return topk_ids, topk_probs
 


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
                
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True)              
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
                                   return_dict = True)                 
        # print(output)
        answer_loss = output.logits
        # print(answer_loss.size(), output.loss, output.logits.shape)
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
    
def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    


'''
CausalLMOutputWithCrossAttentions(loss=tensor(0.0053, device='cuda:0'), logits=tensor([[[ -7.8046,  -8.2725,  -8.1243,  ...,  -8.1776,  -9.1671,  -4.3873],
         [-10.7337, -11.2378, -11.0732,  ...,  -8.5200,  -9.9859, -11.6836],
         [-10.7980, -11.0448, -10.7752,  ...,  -8.5740,  -7.8875,  -7.6834],
         ...,
         [ -7.6959,  -8.1183,  -8.0096,  ...,  -7.8264,  -9.3861,  -4.7072],
         [ -7.7363,  -8.1864,  -8.0704,  ...,  -7.8686,  -9.4454,  -4.8006],
         [ -7.6766,  -8.0966,  -7.9966,  ...,  -7.8886,  -9.3252,  -4.7050]],

        [[-12.1273, -12.0666, -12.5410,  ..., -10.5576, -11.0925,  -5.6486],
         [ -8.4966,  -8.4834,  -8.4985,  ...,  -9.1983,  -7.2372,  -4.4791],
         [-10.5081, -11.1845, -10.9536,  ...,  -8.7220,  -8.5938,  -8.7204],
         ...,
         [-12.1932, -12.1310, -12.6297,  ..., -10.6736, -11.2102,  -5.9023],
         [-12.6169, -12.5828, -13.0535,  ..., -11.0631, -11.6973,  -6.2310],
         [-12.6053, -12.5638, -13.0265,  ..., -10.9752, -11.6520,  -6.1796]],

        [[-12.0144, -11.6965, -12.1704,  ...,  -9.1434, -10.0132,  -7.9319],
         [ -7.1713,  -6.1829,  -6.8806,  ...,  -5.3852,  -6.1873,  -5.7262],
         [ -8.3585,  -8.3241,  -8.0926,  ...,  -6.0979,  -6.9201,  -1.6633],
         ...,
         [-12.4069, -12.0450, -12.6058,  ...,  -9.5720, -10.6156,  -8.4975],
         [-12.5971, -12.2828, -12.8166,  ...,  -9.8528, -10.8201,  -8.8563],
         [-12.7734, -12.5188, -13.0271,  ...,  -9.9447, -10.9536,  -9.0752]],

        ...,

        [[-11.2172, -11.2199, -11.8307,  ..., -10.2309,  -9.4470, -11.9161],
         [ -7.9850,  -8.3715,  -8.2827,  ...,  -7.6591,  -9.2347, -12.2111],
         [-11.5972, -12.5353, -11.8026,  ...,  -9.2272,  -8.3953, -13.5020],
         ...,
         [-11.3080, -11.2831, -11.9278,  ..., -10.3502,  -9.5051, -12.5760],
         [-11.6321, -11.6812, -12.2843,  ..., -10.7200,  -9.8499, -12.6954],
         [-11.6869, -11.7311, -12.3215,  ..., -10.6685,  -9.8518, -12.6453]],

        [[-11.1865, -11.2316, -11.8267,  ..., -10.3126,  -9.3449, -10.2850],
         [ -9.3449,  -9.5861,  -9.4332,  ...,  -9.6910,  -9.0240,  -9.0461],
         [-11.1709, -11.8281, -11.3993,  ...,  -7.8001,  -8.2264, -12.4585],
         ...,
         [-11.3539, -11.3752, -11.9871,  ..., -10.5030,  -9.4343, -10.6106],
         [-11.6314, -11.7168, -12.3153,  ..., -10.8255,  -9.8478, -10.6376],
         [-11.6717, -11.7546, -12.3496,  ..., -10.7898,  -9.8691, -10.6311]],

        [[-11.2224, -11.1903, -11.8596,  ..., -10.1466,  -8.8655,  -9.3798],
         [-11.1376, -11.0230, -11.2865,  ..., -10.9805,  -9.8223,  -9.4653],
         [-11.6906, -12.3470, -11.8356,  ...,  -9.0791,  -8.2954, -10.5326],
         ...,
         [-11.3842, -11.3481, -12.0298,  ..., -10.3992,  -9.1015,  -9.8234],
         [-11.6914, -11.6888, -12.3495,  ..., -10.7024,  -9.4368,  -9.9726],
         [-11.7150, -11.7043, -12.3608,  ..., -10.6137,  -9.4078,  -9.9106]]],
       device='cuda:0'), past_key_values=None, hidden_states=None, attentions=None, cross_attentions=None)
torch.Size([]) tensor(0.0053, device='cuda:0') torch.Size([1024, 9, 30522])
'''