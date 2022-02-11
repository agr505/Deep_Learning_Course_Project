import torch
import logging as log
import torch.nn as nn
from transformers import *
from transformers.modeling_bert import *
import numpy as np
class CustomTAVAT(nn.Module):
    def __init__(self):
        super(CustomTAVAT, self).__init__()

        label_list = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        num_labels = len(label_list)
        config3 = BertConfig.from_pretrained(
        'bert-base-cased',
        num_labels=num_labels,
        )
        self.tavatbert  = BertForTokenClassification.from_pretrained(
        'bert-base-cased',
        from_tf=bool(".ckpt" in 'bert-base-cased'),
        config=config3) 

        self.theta = torch.nn.Parameter(torch.zeros((2,1, 256,768)))  
        #self.theta = torch.nn.Parameter(torch.zeros((2,1, 1,768)))  
       
        

    def forward(self,inputs,batch, embeds_init, delta_lb, delta_tok,trainbool,args):
        if trainbool==True:
            thetasoft=self.theta.softmax(0)
            #print("thetasoft Dimensions ",thetasoft.shape)
            
            #print("thetasoft2 Dimensions ",thetasoft[0,:,:,:].shape)
            inputs_embeds = embeds_init + delta_lb*thetasoft[0,:,:,:] + delta_tok*thetasoft[1,:,:,:]

            
            inputs['inputs_embeds'] = inputs_embeds
        
        
            outputs= self.tavatbert(token_type_ids=None, attention_mask= batch[1],     labels= batch[3], inputs_embeds=inputs_embeds)
        else:

            

            outputs=self.tavatbert(inputs['input_ids'], token_type_ids=None, attention_mask=inputs["attention_mask"])
       
        

        return inputs,outputs