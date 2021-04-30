import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from src.encoding.models import Encoding, DataEncodings

UP_TO = 'up_to'
ONLY_THIS = 'only'
ALL_IN_ONE = 'all_in_one'

# padding
ZERO_PADDING = 'zero_padding'
NO_PADDING = 'no_padding'
PADDING_VALUE = 0

PREFIX_ = 'prefix_'


class Encoder:
    def __init__(self, df: DataFrame, encoding: Encoding):
        self._encoder = {}
        self._label_dict = {}
        self._label_dict_decoder = {}
        self._init_encoder(df, encoding)

    def _init_encoder(self, df: DataFrame, encoding: Encoding):
        #le = LabelEncoder()
        #print("VALORI")
        #print(df.values)

        #print(encoding.data_encoding) #label_encoder
        #print(DataEncodings.LABEL_ENCODER.value)#label_encoder
        #enc = le.fit(np.unique(df.values))
        #print("ENCODER")
        #print(enc.classes_)
        val = [ ]
        for column in df:
            l= df[column].values
            #print(l)
            for index,value in enumerate(l):
                val.append(value)

            #val.append(dati)

        val = np.unique(val)
        val = [str(elem)for elem in val]
        #v = pd.DataFrame(val)
        #print(v)

        #s= sorted(pd.concat([pd.Series([str(PADDING_VALUE)]), v]))

        le =LabelEncoder()
        le.fit(val)
        #print("CLASSI")
        c= le.classes_
        #print(le.classes_)
        #print("TRASFORM")
        #print(le.transform(c))




        for column in df:
            if column != 'trace_id':
                #print(df[column])
                if df[column].dtype != int or (df[column].dtype == int and np.any(df[column] < 0)): # quando anche un
                    # solo valore della colonna è minore di 0
                    # dal tipo int
                    if encoding.data_encoding == DataEncodings.LABEL_ENCODER.value:
                        #passa sempre di qua



                        #self._encoder[column] = LabelEncoder().fit(sorted(pd.concat([pd.Series([str(
                        # PADDING_VALUE)]), df[column].apply(lambda x: str(x))])))

                        self._encoder[column] = le
                        classes = sorted(pd.concat([ pd.Series([ str(PADDING_VALUE) ]),df[column].apply(lambda x: str(x))]))
                        transforms = self._encoder[column].transform(classes)
                        self._label_dict[column] = dict(zip(classes, transforms))
                        self._label_dict_decoder[column] = dict(zip(transforms, classes))
                    elif encoding.data_encoding == DataEncodings.ONE_HOT_ENCODER.value:
                        raise NotImplementedError('Onehot encoder not yet implemented')
                    else:
                        raise ValueError('Please set the encoding technique!')

    def encode(self, df: DataFrame, encoding: Encoding) -> None:
        for column in df:
            if column in self._encoder:
                if encoding.data_encoding == DataEncodings.LABEL_ENCODER.value:
                    df[column] = df[column].apply(lambda x: self._label_dict[column].get(str(x), PADDING_VALUE))
                elif encoding.data_encoding == DataEncodings.ONE_HOT_ENCODER.value:
                    raise NotImplementedError('Onehot encoder not yet implemented')
                else:
                    raise ValueError('Please set the encoding technique!')

    def decode(self, df: DataFrame, encoding: Encoding) -> None:
        for column in df:
            if column in self._encoder:
                if encoding.data_encoding == DataEncodings.LABEL_ENCODER.value:
                    df[column] = df[column].apply(lambda x: self._label_dict_decoder[column].get(x, PADDING_VALUE))
                elif encoding.data_encoding == DataEncodings.ONE_HOT_ENCODER.value:
                    raise NotImplementedError('Onehot encoder not yet implemented')
                else:
                    raise ValueError('Please set the encoding technique!')
