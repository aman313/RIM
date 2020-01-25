from allennlp.data import DatasetReader


@DatasetReader.register('scan_reader')
def ScanDataReader(DatasetReader):
    pass