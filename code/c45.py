import math
import utils


def freq(dataset):
    """ Returns counts of variant _v_
        in column _col_ of table _table_.
    """
    c4=[]
    for r in dataset:
        c = r.split(",")
        cl=int(c[4])
        if cl==2:
            c4.append(r)
        else:
            c4.append(r)
            
    return c4


def info(table, res_col):
    """ Calculates the entropy of the table _table_
        where res_col column = _res_col_.
    """
    s = 0 # sum
    for v in utils.deldup(table[res_col]):
        p = freq(table, res_col, v) / float(len(table[res_col]))
        s += p * math.log(p, 2)
    return -s


def infox(table, col, res_col):
    """ Calculates the entropy of the table _table_
        after dividing it on the subtables by column _col_.
    """
    s = 0 # sum
    for subt in utils.get_subtables(table, col):
        s += (float(len(subt[col])) / len(table[col])) * info(subt, res_col)
    return s
    


def gain(table, x, res_col):
    """ The criterion for selecting attributes for splitting.
    """
    return info(table, res_col) - infox(table, x, res_col)
