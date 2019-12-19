class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, _lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = _lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class total_time():
    def __init__(self, begin=0):
        self.begin = begin
        self.current = begin

    def update(self, val):
        self.current = val

    def __format__(self, format_spec):
        if self.current < self.begin:
            return '0.0min'
        else:
            minute = (self.current - self.begin) / 60
            day = minute // (24 * 60)
            hour = (minute - (day * 24 * 60)) // 60
            minute = minute - (day * 24 * 60) - (hour * 60)
            time_s = ''
            if day > 0:
                time_s = time_s + str(int(day)) + 'd'
            if hour > 0:
                time_s = time_s + str(int(hour)) + 'h'
            time_s = time_s + '{:.1f}'.format(minute) + 'min'
            return time_s
