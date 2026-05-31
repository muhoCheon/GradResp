#!/usr/bin/env python
"""Update docs_my/experiments/group1_validation.md from run metadata files."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_my.tools.make_run_metadata import (
    DATASET_LABELS,
    METHOD_LABELS,
    collect_source_metrics,
    eval_api_paths,
    output_paths,
)


CHECKLIST_BEGIN = '<!-- GROUP1_CHECKLIST:BEGIN -->'
CHECKLIST_END = '<!-- GROUP1_CHECKLIST:END -->'
MAIN_RESULTS_BEGIN = '<!-- GROUP1_MAIN_RESULTS:BEGIN -->'
MAIN_RESULTS_END = '<!-- GROUP1_MAIN_RESULTS:END -->'
EVAL_API_RESULTS_BEGIN = '<!-- GROUP1_EVAL_API_RESULTS:BEGIN -->'
EVAL_API_RESULTS_END = '<!-- GROUP1_EVAL_API_RESULTS:END -->'

CHECKLIST_COLUMNS = [
    'Dataset',
    'Method',
    'GPU',
    'Runtime',
    'main OOD',
    'main FSOOD',
    'eval_api OOD',
    'eval_api FSOOD',
]

METADATA_TO_COLUMN = {
    'dataset': 'Dataset',
    'method': 'Method',
    'gpu': 'GPU',
    'runtime': 'Runtime',
    'main OOD': 'main OOD',
    'main FSOOD': 'main FSOOD',
    'eval_api OOD': 'eval_api OOD',
    'eval_api FSOOD': 'eval_api FSOOD',
    'eval OOD': 'eval_api OOD',
    'eval FSOOD': 'eval_api FSOOD',
}

RESULT_COLUMNS = [
    'Dataset',
    'Method',
    'OOD Near AUROC',
    'OOD Near FPR95',
    'OOD Far AUROC',
    'OOD Far FPR95',
    'FSOOD Near AUROC',
    'FSOOD Near FPR95',
    'FSOOD Far AUROC',
    'FSOOD Far FPR95',
]

RESULT_METRICS = [
    ('OOD', 'Near', 'AUROC'),
    ('OOD', 'Near', 'FPR95'),
    ('OOD', 'Far', 'AUROC'),
    ('OOD', 'Far', 'FPR95'),
    ('FSOOD', 'Near', 'AUROC'),
    ('FSOOD', 'Near', 'FPR95'),
    ('FSOOD', 'Far', 'AUROC'),
    ('FSOOD', 'Far', 'FPR95'),
]

DATASET_SLUGS = {label: slug for slug, label in DATASET_LABELS.items()}
METHOD_SLUGS = {}
for slug, label in METHOD_LABELS.items():
    METHOD_SLUGS.setdefault(label, slug)


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


def source_metrics_present(metadata, source):
    prefix = f'{source} '
    return any(key.startswith(prefix) for key in metadata)


def enrich_source_metrics(metadata):
    dataset = DATASET_SLUGS.get(metadata['dataset'])
    method = METHOD_SLUGS.get(metadata['method'])
    if dataset is None or method is None:
        return metadata

    if not source_metrics_present(metadata, 'main'):
        try:
            ood_csv, fsood_csv = output_paths(Path('results_test'), method, dataset)
            metadata.update(
                collect_source_metrics('main', ood_csv, fsood_csv, require=False))
        except (FileNotFoundError, ValueError):
            pass

    if not source_metrics_present(metadata, 'eval_api'):
        try:
            ood_csv, fsood_csv = eval_api_paths(method, dataset)
            metadata.update(
                collect_source_metrics(
                    'eval_api',
                    ood_csv,
                    fsood_csv,
                    require=False,
                ))
        except (FileNotFoundError, ValueError):
            pass

    return metadata


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


def table_lines(columns, rows):
    lines = [make_table_row(columns), make_separator_row(len(columns))]
    lines.extend(make_table_row(row) for row in rows)
    return lines


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
    ]


def find_marker_block(lines, begin, end):
    try:
        start = lines.index(begin)
        stop = lines.index(end)
    except ValueError as exc:
        raise ValueError(f'marker block not found: {begin} ... {end}') from exc
    if start >= stop:
        raise ValueError(f'invalid marker block: {begin} ... {end}')
    return start, stop


def table_rows_from_block(lines):
    rows = []
    for line in lines:
        if not line.startswith('| '):
            continue
        cells = split_table_row(line)
        if cells[:2] == ['Dataset', 'Method']:
            continue
        rows.append(cells)
    return rows


def normalize_checklist_cells(cells):
    if len(cells) >= 12:
        # Compact any checklist row that includes appended metric columns.
        cells = cells[:8]
    if len(cells) != len(CHECKLIST_COLUMNS):
        raise ValueError(f'unexpected checklist row width {len(cells)}: {cells}')
    return cells


def update_checklist_rows(rows, metas, reset):
    updated = 0
    output = []
    for cells in rows:
        cells = normalize_checklist_cells(cells)
        if reset:
            output.append(reset_row(cells))
            updated += 1
            continue

        metadata = metas.get((cells[0], cells[1]))
        if metadata is None:
            output.append(cells)
            continue

        output.append(update_row_from_metadata(cells, metadata))
        updated += 1
    return output, updated


def format_result_value(metadata, source, split, group, metric):
    key = f'{source} {split} {group} {metric}'
    mean = metadata.get(key, '-')
    std = metadata.get(f'{key} std', '-')
    if mean == '-' and source == 'main' and metric == 'AUROC':
        mean = metadata.get(f'{split} {group} {metric}', '-')
    if mean == '-':
        return '-'
    if std == '-':
        return mean
    return f'{mean} ± {std}'


def result_rows(checklist_rows, metas, source):
    rows = []
    for cells in checklist_rows:
        cells = normalize_checklist_cells(cells)
        key = (cells[0], cells[1])
        metadata = metas.get(key)
        if metadata is None:
            rows.append([cells[0], cells[1]] + ['-'] * len(RESULT_METRICS))
            continue
        values = [
            format_result_value(metadata, source, split, group, metric)
            for split, group, metric in RESULT_METRICS
        ]
        rows.append([cells[0], cells[1]] + values)
    return rows


def replace_block(lines, begin, end, block_lines):
    start, stop = find_marker_block(lines, begin, end)
    return lines[:start + 1] + block_lines + lines[stop:]


def main():
    args = parse_args()
    validation_path = Path(args.validation_md)
    metas = {}

    if not args.reset:
        for path in metadata_paths(args):
            metadata = parse_metadata(path)
            metadata = enrich_source_metrics(metadata)
            key = (metadata['dataset'], metadata['method'])
            metas[key] = metadata

    lines = validation_path.read_text().splitlines()
    checklist_start, checklist_stop = find_marker_block(
        lines, CHECKLIST_BEGIN, CHECKLIST_END)
    checklist_rows = table_rows_from_block(lines[checklist_start + 1:checklist_stop])
    checklist_rows, updated = update_checklist_rows(
        checklist_rows, metas, args.reset)

    lines = replace_block(
        lines,
        CHECKLIST_BEGIN,
        CHECKLIST_END,
        table_lines(CHECKLIST_COLUMNS, checklist_rows),
    )
    if not args.reset:
        lines = replace_block(
            lines,
            MAIN_RESULTS_BEGIN,
            MAIN_RESULTS_END,
            table_lines(RESULT_COLUMNS, result_rows(checklist_rows, metas, 'main')),
        )
        lines = replace_block(
            lines,
            EVAL_API_RESULTS_BEGIN,
            EVAL_API_RESULTS_END,
            table_lines(
                RESULT_COLUMNS,
                result_rows(checklist_rows, metas, 'eval_api'),
            ),
        )

    validation_path.write_text('\n'.join(lines) + '\n')
    print(f'updated rows: {updated}')


if __name__ == '__main__':
    main()
