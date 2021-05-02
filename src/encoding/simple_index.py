import pandas as pd
from pandas import DataFrame
from pm4py.objects.log.log import Trace, EventLog

from src.utils.trace_attibutes import get_prefix_length, get_max_prefix_length
from src.encoding.encoder import PREFIX_
from src.encoding.models import Encoding, TaskGenerationTypes
from src.labelling.common import compute_label_columns, get_intercase_attributes, add_labels
from src.labelling.models import Labelling

ATTRIBUTE_CLASSIFIER = None


def simple_index(log: EventLog, log2: EventLog, labelling: Labelling, encoding: Encoding) -> DataFrame:

    max_prefix_length,max_length = get_max_prefix_length(log, log2, encoding.prefix_length)
    lung_last = max_length - max_prefix_length  #lunghezza del 30%  91
    #print("max_prefix_len")
    #print(int(max_prefix_length))#214
    #print(max_length) #305

    columns = _compute_columns(int(max_prefix_length)) #prefissi colonne delle prime parti + "trace_id"

    normal_columns_number = len(columns)
    #print("normal_col")
    #print(normal_columns_number)#215

    columns = compute_label_columns(columns, encoding, labelling,lenth=0)#aggiungo prefesso per la label

    encoded_data = []
    kwargs = get_intercase_attributes(log, encoding)
for trace in log: #per oggni traccia, quindi per ogni insieme di eventi
        #pre ogni traccia mi ricavo la lunghezza 70%
        prefix_length = get_prefix_length(len(trace), encoding.prefix_length)
        #print(trace.attributes['concept:name'])
        #print("dimensions")
        #print(len(trace))
        #print(prefix_length)
        if len(trace) <= prefix_length - 1 and not encoding.padding:
            # trace too short and no zero padding
            continue
        if encoding.task_generation_type == TaskGenerationTypes.ALL_IN_ONE.value:
            #non passo di qua
            for event_index in range(1, min(prefix_length + 1, len(trace) + 1)):
                encoded_data.append(add_trace_row(lung_last,trace, encoding, labelling, event_index,
                                                  normal_columns_number,
                                                  labelling.attribute_name, **kwargs))
        else:
            encoded_data.append(add_trace_row(lung_last,trace, encoding, labelling, prefix_length,
                                              normal_columns_number,
                                              labelling.attribute_name, **kwargs))
            #print(add_trace_row(trace, encoding, labelling, prefix_length, normal_columns_number,
            #                                  labelling.attribute_name, **kwargs))



    columns_tot= _compute_columns(int(max_prefix_length))
    columns_tot = compute_label_columns(columns_tot, encoding, labelling,lenth=lung_last)

    #Ho l'array totale,e adesso devo separare la prima parte e la seconda parte

    full_df= pd.DataFrame(columns=columns_tot, data=encoded_data)

    first_df = (full_df.iloc[:,range(0,normal_columns_number+1)])
    last_df =full_df.iloc[:,range(normal_columns_number,max_length)]
    trace_df = full_df.iloc[:,0]
    last_df = pd.concat([trace_df,last_df],axis=1)

    #print("SEPARATI")
    #print(first_df)
    #print(last_df)



    return full_df,first_df,last_df

def add_trace_row(lung_last, trace: Trace, encoding: Encoding, labelling: Labelling, event_index: int, column_len: int,
                  attribute_classifier=None, executed_events=None, resources_used=None, new_traces=None):
    #devo prendere l'informazione che mi interessa cioè gli item di ogni traccia/sessione

    #ho tutta la traccia
    #prendo la prima parte e prendo la seconda
    """Row in data frame"""
    trace_row = [trace.attributes['concept:name']] #nome della traccia

    trace_first,trace_last = _trace_prefixes(trace, event_index)# prendo la prima parte e la seconda parte della traccia
    trace_row += trace_first #event_index è il prefisso



    #ho il 70% [circa, non sempre] della traccia  senza label
    #qua aggiungo gli zero al primo 70 per cento
    if encoding.padding or encoding.task_generation_type == TaskGenerationTypes.ALL_IN_ONE.value:
        trace_row += [0 for _ in range(len(trace_row), column_len)]#aggiungo tanti zeri per raggiungere la lun max
        # della traccia
        #len(trace_row) lunghezza riga senza zeri
        #print(trace_row)

    #devo aggiungere gli zero al restante 30%
    if encoding.padding or encoding.task_generation_type == TaskGenerationTypes.ALL_IN_ONE.value:
        trace_last+= [0 for _ in range(len(trace_last), lung_last)]

    trace_row += trace_last
    #print(trace_row)

    #trace_row += add_labels(encoding, labelling, event_index, trace, attribute_classifier=attribute_classifier,
    # executed_events=executed_events, resources_used=resources_used, new_traces=new_traces)
    return trace_row


def _trace_prefixes(trace: Trace, prefix_length: int) -> list:
    """List of indexes of the position they are in event_names
    prefix_length indica l'indice del 70% della traccia

    """
    prefixes_first = []
    prefixes_last = []

    for idx, event in enumerate(trace):
        #print(idx)

        #event ={'concept:name': 'TA080', 'lifecycle:transition': 'complete', 'time:timestamp': datetime.datetime(2011, 10, 4, 21, 58, 40, tzinfo=datetime.timezone(datetime.timedelta(0, 7200))), 'trajectory_n': '2', 'summary': 'partly cloudy', 'temperature': 'warm', 'ta_id': 'TA080', 'day_part': 'evening', 'categories': 'passeggiate in siti storici', 'latitude': '43.773235', 'longitude': '11.254791', 'top10': '0', 'top50': '1', 'top100': '1', 'favored_ranking_number': '11'}
        if idx <prefix_length:
            event_name = event['concept:name']
            prefixes_first.append(event_name)
        if idx >= prefix_length:
            event_name = event[ 'concept:name' ]
            prefixes_last.append(event_name)
    return prefixes_first,prefixes_last


def _compute_columns(prefix_length: int) -> list:
    """trace_id, prefixes, any other columns, label

    """
    return ["trace_id"] + [PREFIX_ + str(i + 1) for i in range(0, prefix_length)]
