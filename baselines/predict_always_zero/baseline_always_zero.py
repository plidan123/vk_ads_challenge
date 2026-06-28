import sys

import pandas as pd
from scipy import stats
from functools import partial


def load_tasks(tasks_filename):
    return pd.read_csv(tasks_filename, sep="\t")


def main():
    tests_filename = sys.argv[1]

    tasks = load_tasks(tests_filename)

    tasks = tasks.assign(
        at_least_one=0,
        at_least_two=0,
        at_least_three=0,
    )
    tasks[['at_least_one', 'at_least_two', 'at_least_three']].to_csv(sys.stdout, sep="\t", index=False, header=True)


if __name__ == '__main__':
    main()
