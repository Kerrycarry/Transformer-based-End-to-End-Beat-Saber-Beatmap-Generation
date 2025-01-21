import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from matplotlib.animation import FuncAnimation
from collections import OrderedDict


def check_same(stride):
    if isinstance(stride, (list, tuple)):
            assert (len(stride) == 2 and stride[0] == stride[1]) or (len(stride) == 3 and stride[0] == stride[1] and stride[1] == stride[2])
            stride = stride[0]
    return stride


def receptive_field(model, input_size, batch_size=-1, device="cuda"):
    '''
    :parameter
    'input_size': tuple of (Channel, Height, Width)
    :return  OrderedDict of `Layername`->OrderedDict of receptive field stats {'j':,'r':,'start':,'conv_stage':,'output_shape':,}
    'j' for "jump" denotes how many pixels do the receptive fields of spatially neighboring units in the feature tensor
        do not overlap in one direction.
        i.e. shift one unit in this feature map == how many pixels shift in the input image in one direction.
    'r' for "receptive_field" is the spatial range of the receptive field in one direction.
    'start' denotes the center of the receptive field for the first unit (start) in on direction of the feature tensor.
        Convention is to use half a pixel as the center for a range. center for `slice(0,5)` is 2.5.
    '''
    def register_hook(module):
        pointwise_operations = ['ReLU', 'LeakyReLU',
                                'ELU', 'Hardshrink', 'Hardsigmoid', 'Hardtanh', 'LogSigmoid', 'PReLU',
                                'ReLU6', 'RReLU', 'SELU', 'CELU', 'GELU', 'Sigmoid', 'SiLU', 'Mish',
                                'Softplus', 'Softshrink', 'Softsign', 'Tanh', 'Tanhshrink', 'Threshold', 'GLU']

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            #ParametrizedConv1d is equvalent to Conv1d
            skip_modules = ["Identity", "EncodecResnetBlock", "EncodecLSTM", "LSTM", "Conv1d", "_WeightNorm", "ParametrizedConv1d"]
            if class_name in skip_modules:
                return
            elif class_name == "EncodecConv1d":
                padding_total = module.padding_total
                padding_right = padding_total // 2
                padding_left = int(padding_total - padding_right)
                module = module.conv
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(receptive_field)
            m_key = "%i" % module_idx
            p_key = "%i" % (module_idx - 1)
            receptive_field[m_key] = OrderedDict()

            if not receptive_field["0"]["conv_stage"]:
                print("Enter in deconv_stage")
                receptive_field[m_key]["j"] = 0
                receptive_field[m_key]["r"] = 0
                receptive_field[m_key]["start"] = 0
            else:
                p_j = receptive_field[p_key]["j"]
                p_r = receptive_field[p_key]["r"]
                p_start = receptive_field[p_key]["start"]

                if class_name in ["Conv1d", "ParametrizedConv1d"]:
                    kernel_size = module.kernel_size[0]
                    stride = module.stride[0]
                    padding = padding_left
                    dilation = module.dilation[0]

                    receptive_field[m_key]["j"] = p_j * stride
                    receptive_field[m_key]["r"] = p_r + (kernel_size - 1) * dilation * p_j
                    receptive_field[m_key]["start"] = p_start + ((kernel_size - 1) / 2 - padding) * p_j

                elif class_name == "Conv2d" or class_name == "MaxPool2d" or class_name == "AvgPool2d" or class_name == "Conv3d" or class_name == "MaxPool3d":
                    kernel_size = module.kernel_size
                    stride = module.stride
                    padding = module.padding

                    if class_name == "AvgPool2d":
                        # Avg Pooling does not have dilation, set it to 1 (no dilation)
                        dilation = 1
                    else:
                        dilation = module.dilation

                    kernel_size, stride, padding, dilation = map(check_same, [kernel_size, stride, padding, dilation])
                    receptive_field[m_key]["j"] = p_j * stride
                    receptive_field[m_key]["r"] = p_r + ((kernel_size - 1) * dilation) * p_j
                    receptive_field[m_key]["start"] = p_start + ((kernel_size - 1) / 2 - padding) * p_j

                elif class_name == "ConvTranspose1d":
                    # For transposed conv, compute the values accordingly
                    stride = module.stride[0]
                    padding = module.padding[0]
                    kernel_size = module.kernel_size[0]
                    dilation = module.dilation[0]

                    receptive_field[m_key]["j"] = p_j / stride
                    receptive_field[m_key]["r"] = p_r + (kernel_size - 1) * dilation * p_j
                    receptive_field[m_key]["start"] = p_start - ((kernel_size - 1) / 2 - padding) * p_j

                elif class_name in pointwise_operations or class_name == "BatchNorm1d" or class_name == "BatchNorm2d" or class_name == "Bottleneck" or class_name == "BatchNorm3d":
                    receptive_field[m_key]["j"] = p_j
                    receptive_field[m_key]["r"] = p_r
                    receptive_field[m_key]["start"] = p_start
                elif class_name == "ConvTranspose2d" or class_name == "ConvTranspose3d":
                    receptive_field["0"]["conv_stage"] = False
                    receptive_field[m_key]["j"] = 0
                    receptive_field[m_key]["r"] = 0
                    receptive_field[m_key]["start"] = 0
                else:
                    raise ValueError(f"Module {class_name} not supported yet")

            receptive_field[m_key]["input_shape"] = list(input[0].size())  # only one
            receptive_field[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                # list/tuple
                receptive_field[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                # tensor
                receptive_field[m_key]["output_shape"] = list(output.size())
                receptive_field[m_key]["output_shape"][0] = batch_size

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
            and not isinstance(module, nn.Linear)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(2, *in_size)).type(dtype) for in_size in input_size]
    else:
        x = Variable(torch.rand(2, *input_size)).type(dtype)

    # create properties
    receptive_field = OrderedDict()
    receptive_field["0"] = OrderedDict()
    receptive_field["0"]["j"] = 1.0
    receptive_field["0"]["r"] = 1.0
    receptive_field["0"]["start"] = 0.5
    receptive_field["0"]["conv_stage"] = True
    receptive_field["0"]["output_shape"] = list(x.size())
    receptive_field["0"]["output_shape"][0] = batch_size
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    # print("------------------------------------------------------------------------------")
    # line_new = "{:>20}  {:>10} {:>10} {:>10} {:>15} ".format("Layer (type)", "map size", "start", "jump", "receptive_field")
    # print(line_new)
    # print("==============================================================================")
    # for layer in receptive_field:
    #     # input_shape, output_shape, trainable, nb_params
    #     assert "start" in receptive_field[layer], layer
    #     if len(receptive_field[layer]["output_shape"]) not in [3, 4, 5]:
    #         raise ValueError(f"Unsupported tensor dimensionality: {len(receptive_field[layer]['output_shape'])}")
    #     line_new = "{:7} {:12}  {:>10} {:>10} {:>10} {:>15} ".format(
    #         "",
    #         layer,
    #         str(receptive_field[layer]["output_shape"][2:]),
    #         str(receptive_field[layer]["start"]),
    #         str(receptive_field[layer]["j"]),
    #         format(str(receptive_field[layer]["r"]))
    #     )
    #     print(line_new)

    # print("==============================================================================")
    # add input_shape
    receptive_field["input_size"] = input_size
    return receptive_field

def receptive_field_for_unit(receptive_field_dict, layer, unit_positions):
    """Utility function to calculate the receptive field for specific units in a layer
        using the dictionary calculated above
    :parameter
        'layer': layer name, should be a key in the result dictionary
        'unit_positions': list of spatial coordinates of the units [(H, W) or (L)]
    ```
    alexnet = models.alexnet()
    model = alexnet.features.to('cuda')
    receptive_field_dict = receptive_field(model, (3, 224, 224))
    receptive_field_for_unit(receptive_field_dict, "8", [(6,6), (7,7)])
    ```
    """
    input_shape = receptive_field_dict["input_size"]
    if layer in receptive_field_dict:
        rf_stats = receptive_field_dict[layer]
        results = []
        for unit_position in unit_positions:
            assert len(unit_position) in [1, 2, 3]
            feat_map_lim = rf_stats['output_shape'][2:]
            if np.any([unit_position[idx] < 0 or
                       unit_position[idx] >= feat_map_lim[idx]
                       for idx in range(len(unit_position))]):
                raise Exception("Unit position outside spatial extent of the feature tensor")
            rf_range = [(rf_stats['start'] + idx * rf_stats['j'] - rf_stats['r'] / 2,
                         rf_stats['start'] + idx * rf_stats['j'] + rf_stats['r'] / 2) for idx in unit_position]
            limit = input_shape[1:]  # Skip batch size
            rf_range = [(max(0, rf_range[axis][0]), min(limit[axis], rf_range[axis][1])) for axis in range(len(unit_position))]
            results.append(rf_range)

        # for i, unit_position in enumerate(unit_positions):
        #     print(f"Receptive field size for layer {layer}, unit_position {unit_position}, is \n {results[i]}")
        return results
    else:
        raise KeyError("Layer name incorrect, or not included in the model.")

def convert_note_code(rf_range, segment_duration_in_quaver, minimum_note, bpm_list, sample_rate, total_frames, total_code):
    # [0, segment_duration_in_quaver*self.minimum_note] 范围内的八分音符，segment_duration_in_quaver*minimum_note是exlucsive
    note_quaver = list(range(segment_duration_in_quaver))
    note_quaver = [x * minimum_note for x in note_quaver]    
    # note_code_map = [[round(x * 60 / bpm * code_rate) for x in note_quaver] for bpm in bpm_list]

    note_code_list_closest_map = []
    for bpm, one_total_frame, one_total_code in zip(bpm_list, total_frames, total_code):
        note_frame = [round(x * 60 / bpm * sample_rate) for x in note_quaver]
        note_frame_map = []
        for i in range(len(note_frame)):
            frame = (note_frame[i], note_frame[i+1]-1) if i in range(len(note_frame) - 1) else (note_frame[i], one_total_frame)
            note_frame_map.append(frame)
        assert note_frame_map[-1][0]<note_frame_map[-1][1]
        note_code_closest_map = []
        one_rf_range = rf_range[:one_total_code]
        j = 0
        for i, (start, end) in enumerate(note_frame_map):
            min_distance = float('inf')  # 初始化最小距离为无穷大
            closest_j = None  # 初始化最接近的区间索引
            x1, y1 = 65, 0.6
            x2, y2 = 260, 1
            ratio = (bpm - x1) / (x2 - x1) * (y2 - y1) + y1
            start_end_mid = (start + end) / 2  # 计算 (start, end) 的中点
            # distance = start_end_mid - start
            # start_end_mid = start + distance*ratio

            while j < len(one_rf_range):
                rf1, rf2 = one_rf_range[j][0]
                rf_mid = (rf1 + rf2) / 2  # 计算 (rf1, rf2) 的中点
                
                # 计算中点的绝对差值并更新最接近的区间
                distance = abs(start_end_mid - rf_mid)
                if distance < min_distance:
                    min_distance = distance
                    closest_j = j
                else:
                    #提前结束
                    break
                j += 1
            note_code_closest_map.append(closest_j)
            #下一轮从本轮结果继续
            j = closest_j
        note_code_list_closest_map.append(note_code_closest_map)
    return  note_code_list_closest_map

