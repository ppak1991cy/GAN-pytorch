import torch


def chose_device():
    cuda_available = torch.cuda.is_available()
    return torch.device("cuda" if cuda_available else "cpu")


class Reporter(object):
    """ Report data recorded in program running """

    def __init__(self, keys, t="report", interval=10, file_path=None):
        assert interval != 0, "Interval can not equal Zero."
        assert file_path is None or file_path is str, "File path must be str or None."
        self.type = t
        self.interval = interval
        self.accumulation = 0
        self.file = file_path

        if keys is list:
            self.keys = keys
        else:
            self.keys = list(keys)
        self.record = dict()
        for k in self.keys:
            self.record[k] = 0

    def recive(self, record):
        if record is not list:
            record = list(record)
        assert len(record) == len(self.keys), "Num of data is not consistent."
        for i, k in enumerate(self.keys):
            self.record[k] += record[i]
        self.accumulation += 1

    def report(self, step):
        msg = "iteration: " + str(step) + "/" + self.type + " --> "
        for k in self.keys:
            m = str(k) + ": " + str(round(self.record[k], 2)) + "| "
            msg += m
        # Output msg
        if self.file is None:
            print(msg)
        else:
            with open(self.file) as f:
                print(msg, file=f)
        # Reset record
        for k in self.record.keys():
            self.record[k] = 0
        self.accumulation = 0

    def run(self, data, step):
        self.recive(data)
        if step % self.interval == 0 and step != 0:
            self.report(step)
