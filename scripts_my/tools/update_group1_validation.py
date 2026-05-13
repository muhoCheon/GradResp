#!/usr/bin/env python
"""Update docs_my/experiments/group1_validation.md from run metadata files."""

import argparse
from pathlib import Path


CHECKLIST_COLUMNS = [
    'Dataset',
    'Method',
    'GPU',
    'Runtime',
    'main OOD',
    'main FSOOD',
    'eval OOD',
    'eval FSOOD',
    'OOD Near AUROC',
    'OOD Far AUROC',
    'FSOOD Near AUROC',
    'FSOOD Far AUROC',
]

METADATA_TO_COLUMN = {
    'dataset': 'Dataset',
    'method': 'Method',
    'gpu': 'GPU',
    'runtime': 'Runtime',
    'main OOD': 'main OOD',
    'main FSOOD': 'main FSOOD',
    'eval OOD': 'eval OOD',
    'eval FSOOD': 'eval FSOOD',
    'OOD Near AUROC': 'OOD Near AUROC',
    'OOD Far AUROC': 'OOD Far AUROC',
    'FSOOD Near AUROC': 'FSOOD Near AUROC',
    'FSOOD Far AUROC': 'FSOOD Far AUROC',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--validation-md',
        default='docs_my/experiments/group1_validation.md',
    )
    parser.add_argument('--metadata', nargs='*')
    parser.add_argument('--runs-root', default='results_test/runs')
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset checklist rows to the initial pending state.',
    )
    return parser.parse_args()


def parse_metadata(path):
    data = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line.startswith('- ') or ': ' not in line:
                continue
            key, value = line[2:].split(': ', 1)
            value = value.strip()
            if value.startswith('`') and value.endswith('`'):
                value = value[1:-1]
            data[key] = value

    required = ['dataset', 'method']
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f'{path} missing keys: {missing}')
    return data


def metadata_paths(args):
    if args.metadata:
        return [Path(path) for path in args.metadata]
    return sorted(Path(args.runs_root).glob('*/*/metadata.md'))


def split_table_row(line):
    cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
    return cells


def make_table_row(values):
    return '| ' + ' | '.join(values) + ' |'


def make_separator_row(width):
    return '|' + '|'.join(['---'] * width) + '|'


def update_row_from_metadata(cells, metadata):
    row = dict(zip(CHECKLIST_COLUMNS, cells))
    for meta_key, column in METADATA_TO_COLUMN.items():
        if meta_key in metadata:
            row[column] = metadata[meta_key]
    return [row[column] for column in CHECKLIST_COLUMNS]


def reset_row(cells):
    dataset, method = cells[0], cells[1]
    eval_status = '-' if dataset == 'MNIST' else 'pending'
    return [
        dataset,
        method,
        '-',
        '-',
        'pending',
        'pending',
        eval_status,
        eval_status,
        '-',
        '-',
        '-',
        '-',
    ]


def main():
    args = parse_args()
    validation_path = Path(args.validation_md)
    metas = {}

    if not args.reset:
        for path in metadata_paths(args):
            metadata = parse_metadata(path)
            key = (metadata['dataset'], metadata['method'])
            metas[key] = metadata

    lines = validation_path.read_text().splitlines()
    updated = 0
    output = []

    for line in lines:
        if line.startswith('| Dataset |'):
            output.append(make_table_row(CHECKLIST_COLUMNS))
            continue
        if line.startswith('|---'):
            output.append(make_separator_row(len(CHECKLIST_COLUMNS)))
            continue
        if not line.startswith('| '):
            output.append(line)
            continue

        cells = split_table_row(line)
        if len(cells) > len(CHECKLIST_COLUMNS) and cells[:3] == ['Dataset', 'Method', 'GPU']:
            cells = cells[:3] + cells[4:]
        elif len(cells) > len(CHECKLIST_COLUMNS):
            cells = cells[:3] + cells[4:]
        if len(cells) > len(CHECKLIST_COLUMNS):
            cells = cells[:len(CHECKLIST_COLUMNS)]

        if len(cells) != len(CHECKLIST_COLUMNS):
            output.append(line)
            continue

        if args.reset:
            output.append(make_table_row(reset_row(cells)))
            updated += 1
            continue

        key = (cells[0], cells[1])
        metadata = metas.get(key)
        if metadata is None:
            output.append(line)
            continue

        new_cells = update_row_from_metadata(cells, metadata)
        output.append(make_table_row(new_cells))
        updated += 1

    validation_path.write_text('\n'.join(output) + '\n')
    print(f'updated rows: {updated}')


if __name__ == '__main__':
    main()
