import torch

def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(dim=0, index=segment_ids, src=data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims) # [n_edge, n_channel, d]
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(dim=0, index=segment_ids, src=data)
    count.scatter_add_(dim=0, index=segment_ids, src=torch.ones_like(data))
    return result / count.clamp(min=1)