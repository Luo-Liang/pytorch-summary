
import torch
import torch.nn as nn
from torch.autograd import Variable
import datetime
from collections import OrderedDict
import numpy as np
import time


class Timer:
    def __init__(self, holder, gpu_specifier):
        self.holder = holder
        self.device = gpu_specifier

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        # _smart_print("[%d] synchronizing..." % dist.get_rank())
        torch.cuda.synchronize(self.device)
        # _smart_print("[%d] synchronized." % dist.get_rank())
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        self.holder.append(self.interval)

def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), iter=1000, x=None, dtypes=None, bucketize=False):
    summary_str = ''
    monitored = []
    for param in model.parameters():
        if param.requires_grad:
            monitored.append(param)
            pass
        pass

    iterator = 0
    def register_hook(tensor):
        t = tensor
        def bwHook(grad):
            current = datetime.datetime.now().timestamp() - summary['last_backward_tick']
            #t = grad
            if t not in summary:
                summary[t] = {}
                summary[t]['backward_tick'] = 0
                pass
            summary[t]['backward_tick'] += current
            summary[t]['size'] = torch.prod(torch.LongTensor(list(t.size()))).item() * 4
            summary['last_backward_tick'] = datetime.datetime.now().timestamp()
            #print("activating")
            return grad

        hooks.append(tensor.register_hook(bwHook))
        pass

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]
    if x is None:
        print("vision model detected?")
        if dtypes == None:
            dtypes = [torch.FloatTensor]*len(input_size)
        # batch_size of 2 for batchnorm
        x = [torch.rand(2, *in_size).type(dtype).to(device=device)
             for in_size, dtype in zip(input_size, dtypes)]
        double_asterisk = False
    else:
        double_asterisk = True
    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    #model.apply(register_hook)
    for p in monitored:
        register_hook(p)
        pass

    # make a forward pass
    # print(x.shape)
    if double_asterisk is False:
        output = model(*x)
        loss = (1-output.mean())        
    else:
        output = model(**x)
        loss  = output["loss"] if isinstance(output, dict) else output[0]
    
    for _ in range(iter):
        summary['last_backward_tick'] = datetime.datetime.now().timestamp()
        loss.backward(loss, retain_graph=True)
        pass
    #print(hooks)
    # remove these hooks
    for h in hooks:
        h.remove()
        pass

    #reverse
    trainable_params = []
    backward_ts = []
    del summary['last_backward_tick']
    for key in summary:
        trainable_params.append(summary[key]['size'])
        backward_ts.append(summary[key]['backward_tick'])
        pass
        
    
    summary_str = ''
    BUCKET_CAP = 25 * 1024 * 1024
    BUCKET_CAP_INIT = 1024 * 1024
    bucket_size = 0
    time = 0
    delayed = []
    if bucketize:
        trainable_params = []
        backward_ts = []
        for key in summary:
            size = summary[key]['size']
            bucket_size += size
            time += summary[key]['backward_tick']
            if bucket_size >= (BUCKET_CAP if len(trainable_params) > 0 else BUCKET_CAP_INIT):
                backward_ts.append(time)
                trainable_params.append(bucket_size)
                bucket_size = 0
                pass
            pass
        backward_ts.append(time)
        trainable_params.append(bucket_size)
        pass
    #print(trainable_params)
    sum_ts = sum(backward_ts)
    backward_ts = [x / sum_ts for x in backward_ts]
    return summary_str, list(reversed(trainable_params)), list(reversed(backward_ts))

def summary_string_huggingface(model, x=None, device=torch.device('cuda:0'), iter=100, bucketize=False):
    summary_str = ''
    monitored = []
    for param in model.parameters():
        if param.requires_grad:
            monitored.append(param)
            pass
        pass

    iterator = 0
    def register_hook(tensor):
        t = tensor
        def bwHook(grad):
            current = datetime.datetime.now().timestamp() - summary['last_backward_tick']
            #t = grad
            if t not in summary:
                summary[t] = {}
                summary[t]['backward_tick'] = 0
                pass
            summary[t]['backward_tick'] += current
            summary[t]['size'] = torch.prod(torch.LongTensor(list(t.size()))).item() * 4
            summary['last_backward_tick'] = datetime.datetime.now().timestamp()
            #print("activating")
            return grad

        hooks.append(tensor.register_hook(bwHook))
        pass
    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    #model.apply(register_hook)
    fw_times = []
    for _ in range(iter):
        with Timer(fw_times, device) as timer:
            output = model(**x)
    fw = np.mean(fw_times)

    loss  = output["loss"] if isinstance(output, dict) else output[0]

    bw_times = []
    for _ in range(iter):
        with Timer(bw_times, device) as timer:
            loss.backward(loss, retain_graph=True)

    bw = np.mean(bw_times)
   
    for p in monitored:
        register_hook(p)
        pass
    

    
    for _ in range(iter):
        summary['last_backward_tick'] = datetime.datetime.now().timestamp()
        loss.backward(loss, retain_graph=True)
        pass
    #print(hooks)
    # remove these hooks
    for h in hooks:
        h.remove()
        pass
    


    #reverse
    trainable_params = []
    backward_ts = []
    del summary['last_backward_tick']
    for key in summary:
        trainable_params.append(summary[key]['size'])
        backward_ts.append(summary[key]['backward_tick'])
        pass
        
    
    summary_str = ''
    BUCKET_CAP = 25 * 1024 * 1024
    BUCKET_CAP_INIT = 1024 * 1024
    bucket_size = 0
    time = 0
    delayed = []
    if bucketize:
        trainable_params = []
        backward_ts = []
        for key in summary:
            size = summary[key]['size']
            bucket_size += size
            time += summary[key]['backward_tick']
            if bucket_size >= (BUCKET_CAP if len(trainable_params) > 0 else BUCKET_CAP_INIT):
                backward_ts.append(time)
                trainable_params.append(bucket_size)
                bucket_size = 0
                pass
            pass
        backward_ts.append(time)
        trainable_params.append(bucket_size)
        pass
    #print(trainable_params)
    sum_ts = sum(backward_ts)
    backward_ts = [x / sum_ts for x in backward_ts]
    return fw, bw ,list(reversed(trainable_params)), list(reversed(backward_ts))