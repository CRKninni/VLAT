import argparse
import os
import  yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_vqa import ALBEF
from models.vit import interpolate_pos_embed
from transformers import BertTokenizer

import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn

from scheduler import create_scheduler
from optim import create_optimizer
import wandb


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, ep_size, global_step=0):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    wandb.define_metric("global_step")
    wandb.define_metric("evaluation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50    
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
    
    for i,(image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)      
        question_input = tokenizer(question, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(device) 
        answer_input = tokenizer(answer, padding='longest', return_tensors="pt").to(device) 
        
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        loss = model(image, question_input, answer_input, train=True, alpha=alpha, k=n, weights=weights)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=lr)

        wandb.log({'train/loss': loss.item(), 'global_step': global_step})
        wandb.log({'train/lr': lr, 'global_step': global_step})
        epoch = global_step / ep_size
        wandb.log({'train/epoch': epoch, 'global_step': global_step})
        
        global_step+=1
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size) 
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []
    
    answer_list = [answer+config['eos'] for answer in data_loader.dataset.answer_list]
    answer_input = tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):        
        image = image.to(device,non_blocking=True)             
        question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)        

        topk_ids, topk_probs = model(image, question_input, answer_input, train=False, k=config['k_test'])      
        
        for ques_id, topk_id, topk_prob in zip(question_id, topk_ids, topk_probs):
            ques_id = int(ques_id.item())          
            _, pred = topk_prob.max(dim=0)
            # _, pred_class = pred.max(1)
            # accuracy = (targets==pred_class).sum() / targets.size(0)
            result.append({"question_id":ques_id, "answer":data_loader.dataset.answer_list[topk_id[pred]]})   

    return result




def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    
    
    #### Dataset #### 
    print("Creating vqa datasets")
    datasets = create_dataset('vqa', config)   
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]
    
    train_loader, test_loader = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train'],config['batch_size_test']],
                                              num_workers=[4,4],is_trains=[True, False], 
                                              collate_fns=[vqa_collate_fn,None]) 
    print(args.text_encoder)
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    model = model.to(device) 
    arg_opt = utils.AttrDict(config['optimizer'])
    # print(arg_opt)
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)          
        
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped   
        
        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
                
            for key in list(state_dict.keys()):
                if 'bert' in key:
                    encoder_key = key.replace('bert.','')         
                    state_dict[encoder_key] = state_dict[key] 
                # intialize text decoder as multimodal encoder (last 6 layers of model.text_encoder)    
                if 'text_encoder' in key:                
                    if 'layer' in key:
                        encoder_keys = key.split('.')
                        layer_num = int(encoder_keys[4])
                        if layer_num<6:
                            del state_dict[key]  
                            continue
                        else:
                            decoder_layer_num = (layer_num-6)
                            encoder_keys[4] = str(decoder_layer_num)
                            encoder_key = '.'.join(encoder_keys)     
                    else:
                        encoder_key = key
                    decoder_key = encoder_key.replace('text_encoder','text_decoder')  
                    state_dict[decoder_key] = state_dict[key]     

                    del state_dict[key]                
                
        msg = model.load_state_dict(state_dict,strict=False)  
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)  

        
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    
    print("Start training")
    start_time = time.time()
    #checkpoint = torch.load("/home/crk/Desktop/albef_ex/output/vqa/checkpoint_09.pth")
    #model.load_state_dict(checkpoint['model'])
    #start_epoch = checkpoint['epoch']
    try:
        global_step = checkpoint['global_step']
    except:
        print("Global step not present in the checkpoint. Initializing manually")
        global_step = 0
        
    for epoch in range(start_epoch, max_epoch):
        
        if start_epoch!=0 and global_step==0:
            global_step = start_epoch * len(train_loader)
        
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)  
        
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, 						len(train_loader),
            			 global_step)  

        if args.evaluate:
            vqa_result = evaluation(model, test_loader, tokenizer, device, config)
            result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d'%epoch)
            
        if utils.is_main_process():               
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")                        
                         
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
                'global_step' : global_step
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  

        # dist.barrier()   
  	
    vqa_result = evaluation(model, test_loader, tokenizer, device, config)
    
    # wandb.log({'evaluation/vqa_result': , 'global_step': global_step}) -> To Do ...
    result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch%d'%epoch)
                     
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml') 
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--resume_run', default=False, type=bool)
    parser.add_argument('--run_id', default='')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    wandb.login()

    if args.resume_run:
        if not args.run_id:
            sys.exit("Error: 'run_id' must be provided if 'resume_run' is True.")
        wandb.init(
            project="VQA",
            config=config,
            id=args.run_id,
            resume="must"
        )
    else:
        print("Starting a new WandB run...")
        wandb.init(
            project="VQA",
            config=config
        )

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)