from typing import Callable

import pandas as pd
from pandas import DataFrame
from pm4py.objects.log.log import Trace, EventLog

from src.utils.trace_attibutes import get_prefix_length, get_max_prefix_length
from src.encoding.encoder import PREFIX_
from src.encoding.models import Encoding, TaskGenerationTypes
from src.encoding.simple_index import compute_label_columns, add_labels, get_intercase_attributes
from src.labelling.models import Labelling

ATTRIBUTE_CLASSIFIER = None


def complex(log: EventLog, log2: EventLog, labelling: Labelling, encoding: Encoding, additional_columns: dict) -> DataFrame:
    return _encode_complex_latest(log, log2, labelling, encoding, additional_columns, _columns_complex, _data_complex)


def last_payload(log: EventLog, log2: EventLog, labelling: Labelling, encoding: Encoding, additional_columns: dict) -> DataFrame:
    return _encode_complex_latest(log, log2, labelling, encoding, additional_columns, _columns_last_payload,
                                  _data_last_payload)


def _encode_complex_latest(log: EventLog, log2: EventLog, labelling: Labelling, encoding: Encoding, additional_columns: dict,
                           column_fun: Callable, data_fun: Callable) -> DataFrame:

    print(additional_columns)
    max_prefix_length, max_length = get_max_prefix_length(log, log2, encoding.prefix_length)

    lung_last = max_length - max_prefix_length
    print(lung_last)


    columns = column_fun(max_prefix_length, additional_columns)
    columns_init=columns.copy()
    columns_init.pop(0)

    #print("columns_init")
    #print(columns_init)

    normal_columns_number = len(columns)
    columns_label = compute_label_columns(columns, encoding, labelling,lenth=0)



    #print(columns)
    encoded_data = []

    kwargs = get_intercase_attributes(log, encoding)
    for trace in log:
        prefix_length = get_prefix_length(len(trace), encoding.prefix_length)
        if len(trace) <= prefix_length - 1 and not encoding.padding:
            # trace too short and no zero padding
            continue
        if encoding.task_generation_type == TaskGenerationTypes.ALL_IN_ONE.value:
            for i in range(1, min(prefix_length + 1, len(trace) + 1)):
                encoded_data.append(
                    _trace_to_row(lung_last, trace, encoding, labelling, i, data_fun, normal_columns_number,
                                  additional_columns=additional_columns,
                                  atr_classifier=labelling.attribute_name, **kwargs))
        else:
            encoded_data.append(
                _trace_to_row(lung_last, trace, encoding, labelling, prefix_length, data_fun, normal_columns_number,
                              additional_columns=additional_columns,
                              atr_classifier=labelling.attribute_name, **kwargs))

            #print(_trace_to_row(lung_last, trace, encoding, labelling, prefix_length, data_fun, normal_columns_number,
            #                    additional_columns=additional_columns,atr_classifier=labelling.attribute_name,
            #                    **kwargs))


    columns_tot= column_fun(max_prefix_length, additional_columns)
    columns_tot= compute_label_columns(columns_tot, encoding, labelling,lenth=lung_last)

    full_df = pd.DataFrame(columns=columns_tot, data=encoded_data)
    print("STAMPA I TRE")
    print(full_df)

    first_df=full_df[columns_label] #prendo da trace_id fino a label
    print(first_df)

    last_df= full_df.drop(columns_init,axis=1)#
    print(last_df)


    return full_df,first_df,last_df


def _columns_complex(prefix_length: int, additional_columns: dict) -> list:
    columns = ['trace_id']
    columns += additional_columns['trace_attributes']
    for i in range(1, prefix_length + 1):
        columns.append(PREFIX_ + str(i))
        for additional_column in additional_columns['event_attributes']:
            columns.append(additional_column + "_" + str(i))
    return columns


def _columns_last_payload(prefix_length: int, additional_columns: dict) -> list:
    columns = ['trace_id']
    i = 0
    for i in range(1, prefix_length + 1):
        columns.append(PREFIX_ + str(i))
    for additional_column in additional_columns['event_attributes']:
        columns.append(additional_column + "_" + str(i))
    return columns

#ata_fun
def _data_complex(trace: Trace, prefix_length: int, additional_columns: dict) -> list:
    """Creates list in form [1, value1, value2, 2, ...]

    Appends values in additional_columns
    """

    data_first = [trace.attributes.get(att, 0) for att in additional_columns['trace_attributes']]
    data_last = []

    #print("prefix")
    #print(prefix_length) #è il 70% della traccia corrente

    #print("trace")
    #print(trace)

    for idx, event in enumerate(trace):#idx parte da zero
        if idx < prefix_length:
            event_name = event["concept:name"] #prendo il nome dell'evento, quindi l'item
            data_first.append(event_name)

            for att in additional_columns['event_attributes']: #per ogni elemento in event_attributes, prendo il
                data_first.append(event.get(att, '0'))

        if idx>=prefix_length:
            event_name = event[ "concept:name" ]
            data_last.append(event_name)

    #print("data")
    #print(data_first)

    #per ogni traccia prendo tutte le informazioni corrispondenti informazioni per il 70%, a cui devo aggiungere
    # anche le labels
    return data_first,data_last

def _data_last_payload(trace: list, prefix_length: int, additional_columns: dict) -> list:
    """Creates list in form [1, 2, value1, value2,]

    Event name index of the position they are in event_names
    Appends values in additional_columns
    """
    data = list()
    for idx, event in enumerate(trace):

        if idx == prefix_length:
            break
        event_name = event['concept:name']
        data.append(event_name)

    # Attributes of last event
    for att in additional_columns['event_attributes']:
        if prefix_length - 1 >= len(trace):
            value = 0
        else:
            value = trace[prefix_length - 1][att]
        data.append(value)
    return data


def _trace_to_row(lung_last, trace: Trace, encoding: Encoding, labelling: Labelling, event_index: int,
                  data_fun: Callable,
                  columns_len: int,
                  atr_classifier=None, executed_events=None, resources_used=None, new_traces=None,
                  additional_columns: dict = None) -> list:

    trace_row = [trace.attributes["concept:name"]]# id trace
    # prefix_length - 1 == index
    trace_first,trace_last= data_fun(trace, event_index, additional_columns)
    #print("OK")

    trace_row += trace_first
    if encoding.padding or encoding.task_generation_type == TaskGenerationTypes.ALL_IN_ONE.value:
        trace_row += [0 for _ in range(len(trace_row), columns_len)]

    if encoding.padding or encoding.task_generation_type == TaskGenerationTypes.ALL_IN_ONE.value:
        trace_last += [0 for _ in range(len(trace_last), lung_last)]

    #trace_row += add_labels(encoding, labelling, event_index, trace, attribute_classifier=atr_classifier,
    # executed_events=executed_events, resources_used=resources_used, new_traces=new_traces)

    trace_row += trace_last


    return trace_row
