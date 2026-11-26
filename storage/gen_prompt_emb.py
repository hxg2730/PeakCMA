import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

class GenPromptEmb(nn.Module):
    def __init__(
        self,
        data_path = 'FRED',
        model_name = "gpt2",
        device = 'cuda:0',
        input_len = 96,
        d_model = 768,
        layer = 12,
        divide = 'train',
        prompt_batch_size = 4,
        precision = 'float32'
    ):  
        super(GenPromptEmb, self).__init__()
        self.data_path = data_path
        self.device = device
        self.input_len =  input_len
        self.model_name = model_name
        self.d_model = d_model
        self.layer = layer
        self.len = self.input_len-1
        self.prompt_batch_size = prompt_batch_size
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        dtype=getattr(torch,precision)
        if str(device).startswith('cpu') and precision!='float32':
            raise ValueError('CPU encoding uses float32; use CUDA for mixed precision')
        self.model = GPT2Model.from_pretrained(model_name,torch_dtype=dtype).to(self.device).eval()
        self.model.requires_grad_(False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _prepare_prompt(self, input_template, in_data, in_data_mark, i, j):
        # Time series value
        values = in_data[i, :, j].flatten().tolist()
        values_str = ", ".join([f"{value:.3f}" for value in values])

        # Last token
        trends = torch.sum(torch.diff(in_data[i, :, j].flatten()))
        trends_str = f"{trends.item():.3f}"
        
        # Date
        if self.data_path in ['FRED', 'ILI']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d}"
        elif self.data_path in ['ETTh1', 'ETTh2', 'ECL']:
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:00"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:00"
        else: # sub-hourly and hourly data
            start_date = f"{int(in_data_mark[i,0,2]):02d}/{int(in_data_mark[i,0,1]):02d}/{int(in_data_mark[i,0,0]):04d} {int(in_data_mark[i,0,4]):02d}:{int(in_data_mark[i,0,5]):02d}"
            end_date = f"{int(in_data_mark[i,self.len,2]):02d}/{int(in_data_mark[i,self.len,1]):02d}/{int(in_data_mark[i,self.len,0]):04d} {int(in_data_mark[i,self.len,4]):02d}:{int(in_data_mark[i,self.len,5]):02d}"

        # Prompt
        in_prompt = input_template.replace("value1, ..., valuen", values_str)
        in_prompt = in_prompt.replace("Trends", trends_str)
        in_prompt = in_prompt.replace("[t1]", start_date).replace("[t2]", end_date)
        # print("in_prompt: ", in_prompt)

        tokenized_prompt = self.tokenizer.encode(in_prompt, return_tensors="pt").to(self.device)
        return tokenized_prompt

    def forward(self, tokenized_prompt):
        with torch.no_grad():
            prompt_embeddings = self.model(tokenized_prompt).last_hidden_state
        return prompt_embeddings

    def generate_embeddings(self, in_data, in_data_mark):
            # Prompt formatting is CPU work; avoid per-value CUDA synchronisation.
            in_data=in_data.detach().cpu()
            in_data_mark=in_data_mark.detach().cpu()
            input_templates = {
                'FRED': "From [t1] to [t2], the values were value1, ..., valuen every month. The total trend value was Trends",
                'ILI': "From [t1] to [t2], the values were value1, ..., valuen every week. The total trend value was Trends",
                'ETTh1': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ETTh2': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ECL': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'ETTm1': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
                'ETTm2': "From [t1] to [t2], the values were value1, ..., valuen every 15 minutes. The total trend value was Trends",
                'Weather2K-R': "From [t1] to [t2], the values were value1, ..., valuen every 3 hours. The total trend value was Trends",
                'ERA5': "From [t1] to [t2], the values were value1, ..., valuen every hour. The total trend value was Trends",
                'Weather': "From [t1] to [t2], the values were value1, ..., valuen every 10 minutes. The total trend value was Trends"
            }

            input_template = input_templates.get(self.data_path, input_templates['FRED'])
            
            result = torch.empty((len(in_data), self.d_model, in_data.shape[2]), device=self.device)
            pending = []
            for i in range(len(in_data)):
                for j in range(in_data.shape[2]):
                    tokens = self._prepare_prompt(input_template, in_data, in_data_mark, i, j)
                    if tokens.shape[1] > self.model.config.n_positions:
                        raise ValueError("Prompt exceeds GPT-2 context; refusing to silently truncate history")
                    pending.append((i,j,tokens[0]))
            pending.sort(key=lambda item:len(item[2]))
            for start in range(0,len(pending),self.prompt_batch_size):
                batch=pending[start:start+self.prompt_batch_size]
                lengths=torch.tensor([len(item[2]) for item in batch],device=self.device)
                ids=torch.full((len(batch),int(lengths.max())),self.tokenizer.pad_token_id,dtype=torch.long,device=self.device)
                mask=torch.zeros_like(ids)
                for k,(_,_,tokens) in enumerate(batch):
                    ids[k,:len(tokens)]=tokens
                    mask[k,:len(tokens)]=1
                with torch.inference_mode():
                    hidden=self.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
                for k,(i,j,_) in enumerate(batch):
                    result[i,:,j]=hidden[k,lengths[k]-1]
            return result
