import csv


def write_as_stats_u(model,path,accuracy,precision,recall,f1,total):
    with open(path, 'a+') as fh:
        file_writer = csv.writer(fh)
        cur_row = [model,accuracy,precision, recall,f1,total]
        file_writer.writerow(cur_row)
    fh.close()
    return
