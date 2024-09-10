from functools import partial
from models.xbert import BertLMHeadModel
from transformers import ViTConfig, ViTModel,BeitConfig, BeitModel
from transformers import BertTokenizer
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertForMaskedLM, BertModel, BertConfig
import torch
import timm
import numpy as np
from torch import nn

import torch.nn.functional as F

torch.cuda.empty_cache()


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
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)

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
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)

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
    def __init__(self, pretrained=True, image_size=224, output_dim=768, init_deit=True):
        super(ImageEncoder, self).__init__()
        # self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k') if pretrained else ViTModel(ViTConfig(image_size=image_size, patch_size=16, embed_dim=768, mlp_ratio=4, qkv_bias=True, layer_norm_eps=1e-6))
        # self.vit_model = ViTModel(ViTConfig(image_size=image_size, patch_size=16, embed_dim=768, mlp_ratio=4, qkv_bias=True, layer_norm_eps=1e-6))
        self.vit_model = VisionTransformer(
            img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        if init_deit:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.vit_model)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.vit_model.load_state_dict(state_dict,strict=False) 
            print(msg)

    def forward(self, images):
        image_embeds = self.vit_model(images) 
        return image_embeds


class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', output_dim=768):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        text_config_encoder = BertConfig(vocab_size=30522, num_hidden_layers=12, encoder_width=768, fusion_layer=6, num_attention_heads=12, hidden_size=768, layer_norm_eps=1e-12, max_position_embeddings=768, hidden_act="gelu", hidden_dropout_prob=0.2, initializer_range=0.02, intermediate_size=3072, attention_probs_dropout_prob=0.2)   
        self.bert = BertModel(config=text_config_encoder)

    def forward(self, text):
        outputs = self.bert(text['input_ids'], attention_mask=text['attention_mask'], return_dict = True, mode = 'text')
        return outputs.last_hidden_state



class VLAT(nn.Module):
    def __init__(self):
        super().__init__()

        embed_dim = 256
        self.momentum = 0.995
        self.queue_size = 65536
        self.mlm_probability = 0.15
        self.distill = True
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.encoder_decoder = EncoderDecoder()
        
        self.image_encoder_m = ImageEncoder()
        self.text_encoder_m = TextEncoder()
        self.encoder_decoder_m = EncoderDecoder()
        
        text_config_encoder = BertConfig(vocab_size=30522, fusion_layer=6,encoder_width=768, num_hidden_layers=12, num_attention_heads=12, hidden_size=768, layer_norm_eps=1e-12, max_position_embeddings=512, hidden_act="gelu", hidden_dropout_prob=0.2, initializer_range=0.02, intermediate_size=3072, attention_probs_dropout_prob=0.2)   
        self.text_encoder_1 = BertModel(config=text_config_encoder)
        self.text_encoder_1_m = BertModel(config=text_config_encoder)
        

        bert_config = BertConfig.from_json_file("/home/gen/crk/VLAT/configs/config_bert.json")
        self.text_encoder_2 = BertForMaskedLM.from_pretrained("bert-base-uncased", config=bert_config)
        self.text_encoder_2_m = BertForMaskedLM.from_pretrained("bert-base-uncased", config=bert_config)

        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 
        
        self.vision_proj = nn.Linear(768, embed_dim)
        self.vision_proj_m = nn.Linear(768, embed_dim)
        self.text_proj = nn.Linear(768, embed_dim)         
        self.text_proj_m = nn.Linear(768, embed_dim)

        self.itm_head = nn.Linear(768, 2) 

        self.temp = nn.Parameter(torch.ones([]) * 0.07)

        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.model_pairs = [[self.image_encoder,self.image_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.text_encoder_1, self.text_encoder_1_m],
                                [self.text_encoder_2, self.text_encoder_2_m],
                                [self.encoder_decoder, self.encoder_decoder_m],
                                [self.vision_proj,self.vision_proj_m],
                                [self.text_proj, self.text_proj_m],
                               ]
        self.copy_params() 



        
    def make_mask(self, batch_size=16, num_heads=8, S=12):
        binary_mask = torch.randint(0, 2, (batch_size * num_heads, S, S), dtype=torch.bool)
        float_mask = binary_mask.float().masked_fill(binary_mask, float('-inf'))
        return float_mask


    def forward(self, image, questions, alpha=0.4):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        image_features = self.image_encoder(image) # (16. 3, 224, 224) -->  (batch_size, num_patches+1, hidden_dim) # (16, 197, 768) # 16*1,51,296*3
        image_atts = torch.ones(image_features.size()[:-1],dtype=torch.long).to(image.device)
        question_features = self.text_encoder(questions) #input_ids-->(16, sequence_length), (16, sequence_length,768) # (batch_size, sequence_length, hidden_dim), (16, 11, 768) #16*8448*3
        image_embeddings, text_embeddings = self.encoder_decoder(image_features, question_features) # need to Mask, with mask and without masking # mcan
        
        image_feat = F.normalize(self.vision_proj(image_embeddings[:,0,:]),dim=-1)  
        text_feat = F.normalize(self.text_proj(text_embeddings[:,0,:]),dim=-1)   


        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.image_encoder(image)                                          
            question_features_m = self.text_encoder_m(questions)

            image_embeddings_m, text_embeddings_m = self.encoder_decoder_m(image_embeds_m, question_features_m)

            image_feat_m = F.normalize(self.vision_proj_m(image_embeddings_m[:,0,:]),dim=-1)  
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()],dim=1)


            text_feat_m = F.normalize(self.text_proj_m(text_embeddings_m[:,0,:]),dim=-1) 
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)          

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        sim_i2t = image_feat @ text_feat_all / self.temp 
        sim_t2i = text_feat @ image_feat_all / self.temp 
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2

        # self._dequeue_and_enqueue(image_feat_m, text_feat_m)
        
        # answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100) 
        output_pos = self.text_encoder_1(attention_mask = questions.attention_mask, 
                                        inputs_embeds = text_embeddings, 
                                        encoder_hidden_states = image_embeddings,
                                        encoder_attention_mask = image_atts,                             
                                        return_dict = True,
                                        mode = 'fusion',)
        with torch.no_grad():
            bs = image.size(0)          
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)
   
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeddings[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeddings[neg_idx])
            text_atts_neg.append(questions.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_embeds_all = torch.cat([text_embeddings, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([questions.attention_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg, image_embeddings],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)


        output_neg = self.text_encoder_1(attention_mask = text_atts_all, 
                                        inputs_embeds = text_embeds_all, 
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,                             
                                        return_dict = True,
                                        mode = 'fusion',)

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        input_ids = questions.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, 30522, image.device, targets=labels,
                                      probability_matrix = probability_matrix) 


        with torch.no_grad():
            logits_m = self.text_encoder_2_m(input_ids = input_ids, 
                                           attention_mask = questions.attention_mask,
                                           encoder_hidden_states = image_embeddings_m,  
                                           encoder_attention_mask = image_atts,      
                                           return_dict = True,
                                           return_logits = True,   
                                          )    
        mlm_output = self.text_encoder_2(input_ids = input_ids, 
                                       attention_mask = questions.attention_mask,
                                       encoder_hidden_states = image_embeddings,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                       labels = labels,   
                                       soft_labels = F.softmax(logits_m,dim=-1),
                                       alpha = alpha
                                      )                           
        loss_mlm = mlm_output.loss        

        return loss_mlm, loss_ita, loss_itm  


    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    


    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
        


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
    

    
    

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
