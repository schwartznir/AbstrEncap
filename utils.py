'''A module with utiltiies for extracting a dataset and converting it to csv.'''
from pathlib import Path
import csv
import jsonlines
import gzip
import shutil

def valid_type(type_, item):
    # A function telling if an item is of a valid integer type'''
    if type_:
        return item.get('_type') == type_

    return True


def jsonl2csv(filepath, type_=None, include=None, exclude=None):
    # Convert a jsonl file to a csv
    with jsonlines.open(filepath) as reader:

        while True:
            first = reader.read()
            if valid_type(first):
                break

        all_headers = set(first.keys())
        headers = set(include) if include else all_headers
        assert headers.issubset(all_headers)
        if exclude:
            exclude = set(exclude)
            assert exclude.issubset(all_headers)
            headers -= set(exclude)

        with Path(Path(filepath).with_suffix('.csv').name).open('w') as f:
            writer = csv.DictWriter(f, extrasaction='ignore', fieldnames=headers)
            writer.writeheader()
            writer.writerow(first)
            for obj in reader:
                if not valid_type(type_, obj):
                    continue
                writer.writerow(obj)
        print('done!')
