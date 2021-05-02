from pm4py.objects.log.log import EventLog


def get_prefix_length(trace_len: int, prefix_length: float) -> int:
    if prefix_length >= 1:
        return round(prefix_length)
    else:
        return round(prefix_length*trace_len)


def get_max_prefix_length(log: EventLog, log2: EventLog, prefix_length: float) -> int:
    if prefix_length > 1:
        return prefix_length

    lengths= [len(trace) for trace in log]
    lengths2 = [ len(trace) for trace in log2 ]
    lengths.extend((lengths2))
    lengths.append(int(prefix_length))

    max_length = max(lengths)

    prefix_lengths = [get_prefix_length(len(trace), prefix_length) for trace in log]
    prefix_lengths2 = [get_prefix_length(len(trace), prefix_length) for trace in log2]
    prefix_lengths.extend(prefix_lengths2)
    prefix_lengths.append(int(prefix_length))
    max_prefix_length = max(prefix_lengths)

    return max_prefix_length,max_length
