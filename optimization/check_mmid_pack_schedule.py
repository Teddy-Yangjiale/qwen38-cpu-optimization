#!/usr/bin/env python3
"""Differential test of packing loops extracted from baseline/candidate source."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def extract(path):
    source = path.read_text()
    start = source.index('            int64_t item = 0;')
    end = source.index('\n\n            ggml_barrier', start)
    block = source[start:end]
    # Remove data operations, retain the actual expert/group/tail loop control.
    start_gather = block.index('                    for (int r = 0; r < 4; ++r)')
    end_gather = block.index('\n', block.index('ggml_quantize_mat_t', start_gather))
    block = block[:start_gather] + '                    trace.push_back({cur_a, 4*g, 4});' + block[end_gather:]
    start_tail = block.index('                    const struct mmid_row_mapping rm')
    end_tail = block.index('\n', block.index('packed_base + (size_t) t * nbw1, ne10);', start_tail))
    block = block[:start_tail] + '                    trace.push_back({cur_a, t, 1});' + block[end_tail:]
    block = '\n'.join(line for line in block.splitlines() if 'char * packed_base' not in line)
    for variable in ['g', 't']:
        lines = block.splitlines()
        matches = [i for i, line in enumerate(lines) if 'for (int64_t ' + variable + ' =' in line]
        if len(matches) != 1:
            raise ValueError('Unexpected loop structure')
        lines.insert(matches[0] + 1, '                    ++visits;')
        block = '\n'.join(lines)
    return block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    code = '#include <array>\n#include <vector>\n#include <cstdint>\n#include <cstdio>\n#include <random>\n'
    for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]:
        code += ('std::vector<std::array<int64_t,3>> ' + name +
                 '(const std::vector<int64_t> & matrix_row_counts, int nth, int ith, int64_t & visits) {\n'
                 'int n_as = matrix_row_counts.size(); std::vector<std::array<int64_t,3>> trace;\n' +
                 extract(path) + '\nreturn trace;\n}\n')
    code += r'''
int main() {
    std::mt19937 rng(20261020);
    int cases = 0;
    int64_t old_visits = 0, new_visits = 0;
    for (int trial = 0; trial < 1100; ++trial) {
        std::vector<int64_t> counts(trial < 1000 ? 3 : 512);
        if (trial < 1000) {
            counts = {trial % 10, (trial / 10) % 10, trial / 100};
        } else {
            for (auto & c : counts) c = rng() % 1025;
        }
        for (int nth : {1, 2, 7, 38}) {
            int64_t covered = 0, expected = 0;
            for (auto c : counts) expected += c;
            for (int ith = 0; ith < nth; ++ith) {
                auto a = baseline(counts, nth, ith, old_visits);
                auto b = candidate(counts, nth, ith, new_visits);
                if (a != b) { fprintf(stderr, "ownership/order mismatch\n"); return 1; }
                for (auto item : b) covered += item[2];
                ++cases;
            }
            if (covered != expected) { fprintf(stderr, "coverage mismatch\n"); return 2; }
        }
    }
    printf("{\"per_thread_comparisons\":%d,\"old_loop_visits\":%lld,\"new_loop_visits\":%lld,\"status\":\"PASS\"}\n",
           cases, (long long) old_visits, (long long) new_visits);
}
'''
    cpp = args.output / 'extracted_schedule.cpp'
    cpp.write_text(code)
    binary = args.output / 'check_schedule'
    commands = [['clang++', '-std=c++17', '-O2', '-fsanitize=address,undefined', str(cpp), '-o', str(binary)], [str(binary)]]
    results = []
    for command in commands:
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
        results.append({'command': command, 'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr})
        if proc.returncode:
            break
    report = {'kind': 'local_synthetic_task_ownership_not_kernel_performance',
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.baseline, args.candidate]},
              'results': results}
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(results[-1]['stdout'], end='')
    if results[-1]['returncode']:
        raise SystemExit(results[-1]['stderr'])


if __name__ == '__main__':
    main()
