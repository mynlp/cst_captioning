
from flask import url_for
from flask_table import Table, Col, LinkCol

import pytablewriter

class SortableTable(Table):
    Id = Col('ID')
    Exp = LinkCol('Exp', 'experiments',
                  url_kwargs=dict(exp='Exp', dataset='Dataset'),
                  url_kwargs_extra=dict(category='all'), attr='Exp')
    Dataset = Col('Dataset', show=False)
    Run = Col('Run')
    Bleu_1 = Col('Bleu_1')
    Bleu_2 = Col('Bleu_2')
    Bleu_3 = Col('Bleu_3')
    Bleu_4 = Col('Bleu_4')
    METEOR = Col('METEOR')
    ROUGE_L = Col('ROUGE_L')
    CIDEr = Col('CIDEr')
    Perplexity = Col('Perplexity')
    allow_sort = True

    def __init__(self, items, exp, dataset, **kwargs):
        super(SortableTable, self).__init__(items, **kwargs)
        self.url_name = 'leaderboard'
        self.exp = exp
        self.dataset = dataset

    def sort_url(self, col_key, reverse=False):
        if reverse:
            direction = 'desc'
        else:
            direction = 'asc'
        return url_for(self.url_name, sort=col_key, direction=direction,
                       exp=self.exp, dataset=self.dataset)
    
    def save_markdown(self):
        metrics = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr']
        writer = pytablewriter.MarkdownTableWriter()
        writer.table_name = 'results'
        writer.header_list = ['Run'] + metrics
        values = []
        
        for item in self.items:
            item_dict = vars(item)
            row_values = []
            for header in writer.header_list:
                value = item_dict[header]
                if type(value) is float:
                    value = round(value, 3)
                row_values.append(value)
            values.append(row_values)

        writer.value_matrix = values
        writer.write_table()

class Item(object):

    def __init__(self, no, exp, dataset, run, info):
        self.Id = no
        self.Exp = exp
        self.Dataset = dataset
        self.Run = run
        for key in info:
            setattr(self, key, info[key])

        for k in SortableTable.__dict__.keys():
            if isinstance(
                    SortableTable.__dict__[k],
                    Col) and k not in self.__dict__:
                setattr(self, k, 'n/a')
