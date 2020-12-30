
import torch
import torch.nn as nn
from torch.autograd import Variable
import datetime
from collections import OrderedDict
import numpy as np
import time
import os

if 'INSTRUMENT_NO_BACKWARD_TS' in os.environ:
    NO_BACKWARD_TS=True
else:
    NO_BACKWARD_TS=False

if 'INSTRUMENT_NO_BACKWARD' in os.environ:
    NO_BACKWARD=True
else:
    NO_BACKWARD=False





class Timer:
    def __init__(self, holder, gpu_specifier):
        self.holder = holder
        self.device = gpu_specifier

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        # _smart_print("[%d] synchronizing..." % dist.get_rank())
        torch.cuda.synchronize()
        # _smart_print("[%d] synchronized." % dist.get_rank())
        self.end = time.perf_counter()
        self.interval = self.end - self.start
        self.holder.append(self.interval)

def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info

def summary_string_dlrm(model, inputBatch, device, loss_fn_wrap, iter, optimizer, bucketize=False):
    def unpack_batch(b):
        # Experiment with unweighted samples
        return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None

    # X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
    X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)

    def dlrm_wrap(dlrm, X, lS_o, lS_i, device):
        if True:  # .cuda()
            # lS_i can be either a list of tensors or a stacked tensor.
            # Handle each case below:
            if True:
                lS_i = (
                    [S_i.to(device) for S_i in lS_i]
                    if isinstance(lS_i, list)
                    else lS_i.to(device)
                )
                lS_o = (
                    [S_o.to(device) for S_o in lS_o]
                    if isinstance(lS_o, list)
                    else lS_o.to(device)
                )
        return dlrm(X.to(device), lS_o, lS_i)

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
            data_type = t.dtype
            assert data_type == torch.float
            summary[t]['size'] = torch.prod(torch.LongTensor(list(t.size()))).item() * (8 if t.dtype == torch.long else 4)
            summary['last_backward_tick'] = datetime.datetime.now().timestamp()
            #print("activating")
            return grad

        hooks.append(tensor.register_hook(bwHook))
        pass
    
    monitored = []
    for param in model.parameters():
        if param.requires_grad:
            monitored.append(param)
            pass
        pass

    summary = OrderedDict()
    hooks = []

    # register hook
    #model.apply(register_hook)
    fw_times = []
    bw_times = []

    for _ in range(iter):
        with Timer(fw_times, device) as timer:
            Z = dlrm_wrap(model, X, lS_o, lS_i, device)
            E = loss_fn_wrap(Z, T, True, device)
            #warmup
            optimizer.step()
            model.zero_grad()

    
    for i in range(iter):
        with Timer(bw_times, device) as timer:
            print(f"running backward pass at {i}")
            E.backward(retain_graph=True)

    fw = np.mean(fw_times[5:])

    bw = np.mean(bw_times[5:])

    for p in monitored:
        register_hook(p)
        pass
    #drop half of it, because already used in bw aboves
    #E_versions = E_versions[iter:]
    #print(f"done backward pass. dropped {iter} samples . now at {len(E_versions)}")
    for _ in range(iter):
        summary['last_backward_tick'] = datetime.datetime.now().timestamp()
        E.backward(retain_graph=True)
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
            data_type = t.dtype
            assert data_type == torch.float
            summary[t]['size'] = torch.prod(torch.LongTensor(list(t.size()))).item() * (8 if t.dtype == torch.long else 4)
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

def summary_string_huggingface(model, x, optimizer, max_grad_norm, device=torch.device('cuda:0'), iter=100, bucketize=False):

    for key in x:
        assert x[key].requires_grad is False
        x[key].requires_grad = False
    
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
            data_type = t.dtype
            assert data_type == torch.float
            summary[t]['size'] = torch.prod(torch.LongTensor(list(t.size()))).item() * (8 if t.dtype == torch.long else 4)
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

    for _ in range(5):
        output = model(**x)
        optimizer.step()
        model.zero_grad()
        #warmup

    loss  = output["loss"] if isinstance(output, dict) else output[0]
    loss_detached = loss.detach()
    for _ in range(5):
        loss.backward(loss_detached, retain_graph=True)

    with Timer(fw_times, device) as timer:
        for _ in range(iter):
            model(**x)
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
            optimizer.step()
            model.zero_grad()
            #output = output.detach()

    fw = np.mean(fw_times)/iter

    if NO_BACKWARD is True:
        print(f"backward is disabled to save memory. data is false.")
        return fw, None, None, None    

    loss  = output["loss"] if isinstance(output, dict) else output[0]

    bw_times = []
    with Timer(bw_times, device) as timer:
        for _ in range(iter):
            loss.backward(loss_detached, retain_graph=True)

    bw = np.mean(bw_times)/iter

    if NO_BACKWARD_TS is True:
        print(f"backward ts is disabled to save memory. data is false.")
        return fw, bw, None, None
   
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
