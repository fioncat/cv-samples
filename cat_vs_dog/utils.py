#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import matplotlib.pyplot as plt


class ProcessBar(object):

    def __init__(self, max_steps: int, bar_length=70, done_msg=None):
        self.max_steps = max_steps
        self.step = 0
        self.bar_length = bar_length
        self.done_msg = done_msg

    def show_process(self):

        if self.step >= self.max_steps - 1:
            n_arrows = self.bar_length
            n_lines = 0
            percent = 100.0
        else:
            n_arrows = int(self.step * self.bar_length / self.max_steps)
            n_lines = self.bar_length - n_arrows
            percent = self.step * 100.0 / self.max_steps

        n_head = 0 if n_arrows == 0 else 1

        bar = '|' + '=' * (n_arrows - 1) + n_head * '>' + \
              '.' * n_lines + '| ' +\
              '%.2f' % percent + '%' + '\r'

        sys.stdout.write(bar)
        sys.stdout.flush()

        self.step += 1

        if self.step >= self.max_steps:
            print('')
            if self.done_msg is not None:
                print(self.done_msg)


def sample(data, step):
    return [data[idx] for idx in range(0, len(data), step)]


def plot_metrics(train, valid, title, xlabel, ylabel, step):

    train = sample(train, step)
    valid = sample(valid, step)

    x = list(range(len(train)))

    plt.plot(x, train, alpha=0.6, label='Train')
    plt.plot(x, valid, alpha=0.6, label='Validation')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()
    plt.show()
