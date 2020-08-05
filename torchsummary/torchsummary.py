
import torch
import torch.nn as nn
from torch.autograd import Variable
import datetime
from collections import OrderedDict
import numpy as np


def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''
    moduleKeyLookup = {}
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = "%s-%i" % (class_name, module_idx + 1)
            moduleKeyLookup[module] = m_key
            #print("registering")
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
            pass

        def bwHook(module, input, output):
            m_key = moduleKeyLookup[module] #"%s-%i" % (class_name, module_idx + 1)
            summary[m_key]['backward_tick'] = datetime.datetime.now().timestamp() - summary['last_backward_tick']
            summary['last_backward_tick'] = datetime.datetime.now().timestamp()
            #print("activating")
            pass

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))
            hooks.append(module.register_backward_hook(bwHook))
            pass

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    output = model(*x)
    #make a few backward pass too
    loss = (1-output.mean())
    summary['last_backward_tick'] = datetime.datetime.now().timestamp()
    loss.backward()
    #print(hooks)
    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = []
    backward_ts = []
    for layer in summary:
        if layer == 'last_backward_tick':
            #a hack for recording last time.
            continue
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += [int(summary[layer]["nb_params"])]
                backward_ts += [summary[layer]['backward_tick'] * 1000000]
                pass
            pass
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    #summary_str += "Total params: {0:,}".format(total_params) + "\n"
    #summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    #summary_str += "Non-trainable params: {0:,}".format(total_params -
    #trainable_params) + "\n"
    #summary_str += "----------------------------------------------------------------" + "\n"
    #summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    #summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    #summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    #summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    #summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    #min_ts = min(backward_ts)
    #for i in range(len(backward_ts)):
    #    backward_ts[i] -= min_ts
    #    pass
    sum_ts = sum(backward_ts)
    return summary_str, trainable_params, [x / sum_ts for x in backward_ts]
