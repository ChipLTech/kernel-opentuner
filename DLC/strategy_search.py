#!/usr/bin/env python3
"""R4 measured whole-policy search and independent Qwen3-8B CI confirmation.

Search core: hardware-accepted R4 archive SHA256
8735890e6b5fbd635add527d85d6348322b4da03c7e9f1ca8cba685c54aeab52.
Only the CI evaluator/entrypoints are adapted; no node reset is performed.
"""

import argparse
import ast
import csv
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict

DEFAULT_POLICY = ("", "", "", "", "", "1.0", "all", "-1.0", "-1.0", "15", "0", "", "0")

def digest(data):
    return hashlib.sha256(data).hexdigest()



def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)



def atomic_bytes(path, data):
    """Keep existing mmap readers on their old inode during library updates."""
    path = path.resolve()
    mode = stat.S_IMODE(path.stat().st_mode)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name,
                                     delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(data)
            output.flush()
            temporary.chmod(mode)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)



def policy_rows(data):
    rows = [r for r in csv.reader(io.StringIO(data.decode())) if r]
    if not rows or rows[0][0] != 'Test':
        raise ValueError('Expected policy header Test')
    width = len(rows[0])
    result = {}
    for row in rows[1:]:
        if len(row) != width:
            raise ValueError('Invalid policy row: ' + row[0])
        # PickAutoTuneStrategy.cmake overwrites the variable for every match.
        # The last occurrence is the policy that the compiler actually uses.
        result[row[0]] = tuple(row)
    return result



def policy_identity(data):
    """Cache by the header and effective rows read by CMake, retaining raw CSV."""
    header = next(row for row in csv.reader(io.StringIO(data.decode())) if row)
    return digest(json.dumps([header, sorted(policy_rows(data).items())],
                             separators=(',', ':')).encode())



def policy_header(data):
    return next(row for row in csv.reader(io.StringIO(data.decode())) if row)



def replace_rows(data, replacements):
    """Replace whole rows, preserving every other byte, including blank lines."""
    original = policy_rows(data)
    if replacements.keys() - original.keys():
        raise ValueError('Replacement introduces an unknown kernel')
    source_lines = data.splitlines(keepends=True)
    last_occurrence = {}
    for index, line in enumerate(source_lines):
        row = next(csv.reader([line.decode()]), [])
        if row:
            last_occurrence[row[0]] = index
    lines = []
    for index, line in enumerate(source_lines):
        row = next(csv.reader([line.decode()]), [])
        if row and row[0] in replacements and index == last_occurrence[row[0]]:
            replacement = replacements[row[0]]
            if len(replacement) != len(row) or replacement[0] != row[0]:
                raise ValueError('Replacement changes row identity or width')
            if tuple(row) != tuple(replacement):
                output = io.StringIO()
                ending = '\r\n' if line.endswith(b'\r\n') else '\n'
                csv.writer(output, lineterminator=ending).writerow(replacement)
                line = output.getvalue().encode()
        lines.append(line)
    return b''.join(lines)



def kernel_family(name):
    """Normalize only complete dtype/shape tokens, without a kernel allowlist."""
    tokens = []
    for token in name.lower().split('_'):
        if re.fullmatch(r'(bf|fp|f|i|u|int|uint)(8|16|32|64)', token):
            continue
        tokens.append('N' if token.isdecimal() else token)
    return '_'.join(tokens)



def summarize_profiles(paths):
    pattern = re.compile(r'kernel_name:\s+(\S+)\s+cycles:\s+(\d+)')
    samples = defaultdict(list)
    for path in paths:
        with path.open(errors='replace') as source:
            for line in source:
                match = pattern.search(line)
                if match:
                    samples[match[1].removeprefix('custom_')].append(int(match[2]))
    if not samples:
        raise ValueError('Empty kernel profile')
    return {name: {'calls': len(values), 'median': statistics.median(values)}
            for name, values in samples.items()}



def cycle_guidance(base, donor, baseline_stats, donor_stats, minimum_calls=32):
    """Return row gains and automatic family recipes, using profiles as hints."""
    before, after = policy_rows(base), policy_rows(donor)
    gains, groups = {}, defaultdict(list)
    for name in before.keys() & after.keys():
        if before[name] == after[name]:
            continue
        groups[kernel_family(name)].append(name)
        if name not in baseline_stats or name not in donor_stats:
            continue
        a, b = baseline_stats[name], donor_stats[name]
        if min(a['calls'], b['calls']) < minimum_calls:
            continue
        if abs(b['calls'] / a['calls'] - 1) > 0.05:
            continue
        gains[name] = a['calls'] * (a['median'] - b['median'])
    recipes = {}
    for family, names in sorted(groups.items()):
        recipes[family] = (
            {},
            {name: after[name] for name in names},
            {name: after[name] for name in names
             if name in gains and gains[name] > baseline_stats[name]['calls'] * baseline_stats[name]['median'] * 0.003},
        )
    return gains, recipes



def beam_search(initial, seeds, neighbors, measure, priority, *, width=2,
                rounds=3, fanout=6, budget=18, on_round=None, identity=None):
    """Keep multiple measured paths, including regressions needed for interactions.

    measure(state) returns throughput or None for a failed candidate. The caller
    owns materialization, compilation, validation and identity-bound caching.
    """
    scores, visited = {}, set()
    identify = identity or (lambda state: state)

    def visit(state):
        key = identify(state)
        if key not in visited and len(scores) < budget:
            value = measure(state)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError('Invalid candidate score')
            scores[state] = value
            visited.add(key)

    def ranked(states):
        return sorted((s for s in states if scores.get(s) is not None),
                      key=lambda s: (-scores[s], s))

    visit(initial)
    for state in seeds:
        visit(state)
    beam = ranked(scores)[:width]
    for depth in range(rounds):
        queues = [iter(sorted(set(neighbors(parent)), key=lambda s: (-priority(s), s))) for parent in beam]
        proposed, seen = [], set(visited)
        # Give each measured path a share of the budget; absolute cycle hints
        # from a slower parent must not suppress the measured winner's children.
        while queues and len(proposed) < fanout:
            active = []
            for queue in queues:
                for state in queue:
                    key = identify(state)
                    if key not in seen:
                        proposed.append(state)
                        seen.add(key)
                        active.append(queue)
                        break
                if len(proposed) >= fanout:
                    break
            queues = active
        for state in proposed:
            visit(state)
        best = ranked(scores)
        # Keep the winner separately from a fresh exploration path. Reusing
        # only the best parents prevents crossing two successive regressions.
        beam = best[:1]
        beam += [s for s in ranked(proposed) if s not in beam][:max(width-1, 0)]
        if len(beam) < width:
            beam += [s for s in best if s not in beam][:width-len(beam)]
        if on_round:
            on_round(depth, beam, scores)
        if len(scores) >= budget or not proposed:
            break
    best = ranked(scores)[:width]
    if not best:
        raise ValueError('Every candidate failed')
    return best, scores



def search_policy(base, donor, baseline_stats, donor_stats, measure, *,
                  budget=18, rounds=3, on_round=None, confirm=None):
    """Search categorical whole-row options from one or more complete donors.

    measure(policy_bytes) returns full-workload throughput or None. Profiles
    prioritize exploration; they never remove a differing policy from it.
    """
    donors = [donor] if isinstance(donor, bytes) else list(donor)
    profiles = [donor_stats] if isinstance(donor_stats, dict) else list(donor_stats)
    if not donors or len(donors) != len(profiles):
        raise ValueError('Each donor requires a matching profile dictionary')
    before, header = policy_rows(base), policy_header(base)
    after = [policy_rows(data) for data in donors]
    if any(rows.keys() != before.keys() or policy_header(data) != header
           for data, rows in zip(donors, after)):
        raise ValueError('Baseline and donors must describe the same header and kernel set')
    names = sorted(name for name in before if any(rows[name] != before[name] for rows in after))
    options = [[before[name], *sorted({rows[name] for rows in after} - {before[name]})]
               for name in names]
    families = sorted({kernel_family(name) for name in names})
    indices = {name: i for i, name in enumerate(names)}
    groups = [[indices[n] for n in names if kernel_family(n) == family] for family in families]
    samples = defaultdict(list)
    for data, rows, profile in zip(donors, after, profiles):
        gains, _ = cycle_guidance(base, data, baseline_stats, profile)
        for name, gain in gains.items():
            samples[(name, rows[name])].append(gain)
    hints = {key: statistics.median(values) for key, values in samples.items()}
    donor_states = [tuple(options[i].index(rows[name]) for i, name in enumerate(names))
                    for rows in after]
    initial = (0,) * len(names)
    auto = tuple(max(range(len(rows)), key=lambda option: hints.get((name, rows[option]), 0))
                 for name, rows in zip(names, options))

    def materialize(state):
        return replace_rows(base, {name: options[i][option]
                                   for i, (name, option) in enumerate(zip(names, state)) if option})

    def neighbors(state):
        for group in groups:
            for recipe in [initial, *donor_states, auto]:
                child = list(state)
                for i in group:
                    child[i] = recipe[i]
                if tuple(child) != state:
                    yield tuple(child)
        for i in range(len(names)):
            for option in range(len(options[i])):
                if option != state[i]:
                    child = list(state)
                    child[i] = option
                    yield tuple(child)

    def priority(state):
        uncertainty = max((abs(v) for v in hints.values()), default=1)
        return sum(hints.get((name, options[i][option]), -uncertainty)
                   for i, (name, option) in enumerate(zip(names, state)) if option)

    family_rank = sorted(range(len(families)),
                         key=lambda i: (-sum(hints.get((names[j], options[j][auto[j]]), 0)
                                             for j in groups[i]), families[i]))
    seeds = [*donor_states, auto]
    for count in (1, 2, 4):
        chosen = {i for f in family_rank[:count] for i in groups[f]}
        seeds.append(tuple(auto[i] if i in chosen else 0 for i in range(len(names))))

    def report_round(depth, beam, scores):
        if on_round:
            on_round({'round': depth, 'families': families, 'names': names,
                      'option_counts': [len(rows) for rows in options],
                      'beam': [list(s) for s in beam],
                      'scores': [{'state': list(s), 'total': value,
                                  'strategy_sha': digest(materialize(s))}
                                 for s, value in scores.items()]})

    beam, scores = beam_search(initial, seeds, neighbors,
                              lambda state: measure(materialize(state)), priority,
                              budget=budget, rounds=rounds, on_round=report_round,
                              identity=lambda state: policy_identity(materialize(state)))
    winner, ranking, finalists = beam[0], scores[beam[0]], []
    if confirm:
        states = sorted((s for s in scores if scores[s] is not None),
                        key=lambda s: (-scores[s], s))[:3]
        for state in states:
            totals = []
            for _ in range(3):
                total = confirm(materialize(state))
                totals.append(total)
                if total is None or not math.isfinite(total) or total <= 0:
                    break
            valid = len(totals) == 3 and all(t is not None and math.isfinite(t) and t > 0 for t in totals)
            finalists.append({'state': list(state), 'strategy_sha': digest(materialize(state)),
                              'totals': totals, 'median': statistics.median(totals) if valid else None})
        valid = [f for f in finalists if f['median'] is not None]
        if not valid:
            raise RuntimeError('No finalist completed its repeat measurements')
        best = max(valid, key=lambda f: f['median'])
        winner, ranking = tuple(best['state']), best['median']
    return {'strategy': materialize(winner), 'screen_total': scores[winner],
            'ranking_total': ranking, 'finalists': finalists,
            'families': families, 'names': names, 'state': list(winner),
            'row_options': options,
            'candidate_sha': digest(materialize(winner)), 'auto_selected': True}


def confirmation_report(pairs):
    """R4 gate, with complete finite samples from one device epoch."""
    if len(pairs) != 5:
        raise ValueError('Confirmation protocol requires five pairs')
    if len({p.get('recovery_epoch', 0) for p in pairs}) != 1:
        raise ValueError('All five confirmation pairs must share one epoch')
    if any(not math.isfinite(p[k]) or p[k] <= 0
           for p in pairs for k in ('reference', 'candidate')):
        raise ValueError('Confirmation throughput must be finite and positive')
    deltas = [(p['candidate'] / p['reference'] - 1) * 100 for p in pairs]
    mean_gain = statistics.mean(deltas)
    margin = 2.776 * statistics.stdev(deltas) / math.sqrt(5)
    reference_mean = statistics.mean(p['reference'] for p in pairs)
    candidate_mean = statistics.mean(p['candidate'] for p in pairs)
    return {'passed': candidate_mean > reference_mean and mean_gain - margin > 0
            and sum(d > 0 for d in deltas) >= 4,
            'pairs': pairs, 'reference_mean': reference_mean, 'candidate_mean': candidate_mean,
            'mean_gain_pct': mean_gain, 'ci95_pct': [mean_gain - margin, mean_gain + margin]}


def confirm_pairs(run, reference, candidate, count=5, on_pair=None):
    if count != 5:
        raise ValueError('Confirmation requires five pairs')
    # Restart the entire group if a future evaluator changes epochs, not just
    # the interrupted pair. The CI evaluator deliberately never resets a node.
    for group in range(3):
        pairs = []
        epoch = None
        for index in range(count):
            order = [('reference', reference), ('candidate', candidate)]
            if index % 2:
                order.reverse()
            results = {label: run(data, f'pair-{index+1}-{label}-group-{group+1}')
                       for label, data in order}
            epochs = {r.get('recovery_epoch', 0) for r in results.values()}
            if len(epochs) != 1 or (epoch is not None and epochs != {epoch}):
                break
            epoch = epochs.pop()
            pairs.append({'pair': index + 1, 'reference': results['reference']['total'],
                          'candidate': results['candidate']['total'], 'recovery_epoch': epoch})
            for label, result in results.items():
                if result.get('measurement_sha'):
                    pairs[-1][label + '_measurement'] = {
                        'path': str(Path(result['path']) / 'measurement.json'),
                        'sha': result['measurement_sha']}
            if on_pair:
                on_pair(pairs)
        else:
            return pairs
    raise RuntimeError('Confirmation group repeatedly crossed device recovery')


def select_seed_policies(directory, reference):
    """Automatically choose only actual measured legacy-tuner input policies.

    strategy_iter*.csv is a postprocessed per-kernel mixture, not the policy
    which produced that iteration's throughput. It is deliberately ignored.
    Missing new rows use the compiler's existing default-policy semantics.
    """
    header, before = policy_header(reference), policy_rows(reference)
    records = []
    for path in sorted(directory.rglob('executed_strategy.csv')):
        if not re.fullmatch(r'profile_iter\d+', path.parent.name):
            continue
        try:
            metrics = json.loads((path.parent / 'throughput_metrics.json').read_text())
            total = float(metrics['total_tokens_per_s'])
            data = path.read_bytes()
            rows = policy_rows(data)
            if (metrics.get('error') or not math.isfinite(total) or total <= 0
                    or policy_header(data) != header or before.keys() - rows.keys()):
                continue
        except (OSError, ValueError, KeyError, UnicodeError):
            continue
        records.append({'source': str(path), 'data': data, 'total': total,
                        'source_sha': digest(data)})
    names = set(before).union(*(policy_rows(r['data']).keys() for r in records))

    def complete(data):
        missing = names - policy_rows(data).keys()
        if missing and len(header) != len(DEFAULT_POLICY) + 1:
            raise ValueError('New kernel rows require the DLC default-policy schema')
        output = io.StringIO()
        csv.writer(output, lineterminator='\n').writerows(
            (name, *DEFAULT_POLICY) for name in sorted(missing))
        return data + (b'\n' if missing and not data.endswith(b'\n') else b'') + output.getvalue().encode()

    reference = complete(reference)
    seen = {policy_identity(reference)}
    selected = []
    for record in sorted(records, key=lambda r: (-r['total'], r['source'])):
        record['data'] = complete(record['data'])
        identity = policy_identity(record['data'])
        if identity not in seen:
            seen.add(identity)
            selected.append(record)
    if len(selected) < 2:
        raise ValueError('Need two distinct measured policies excluding the confirmation control')
    manifest = [{k: v for k, v in r.items() if k != 'data'}
                | {'sha': digest(r['data']), 'role': 'base' if i == 0 else 'donor'}
                for i, r in enumerate(selected)]
    return reference, selected[0]['data'], [r['data'] for r in selected[1:]], manifest


def prepare_inputs(args):
    reference, base, donors, manifest = select_seed_policies(
        args.seed_dir, args.original_policy.read_bytes())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'reference.csv').write_bytes(reference)
    (args.output / 'base.csv').write_bytes(base)
    for index, data in enumerate(donors):
        (args.output / f'donor-{index+1}.csv').write_bytes(data)
    atomic_json(args.output / 'manifest.json', {
        'reference_sha': digest(reference), 'base_sha': digest(base),
        'donor_shas': [digest(data) for data in donors], 'measured_inputs': manifest,
        'original_strategy_sha': digest(args.original_policy.read_bytes()),
        'original_library_sha': digest(args.original_library.read_bytes())})


class IntegrityFailure(Exception):
    """Source/policy/library interference is fatal, not a bad candidate."""


class CleanupFailure(Exception):
    """An owned worker group remains alive; do not install another library."""


def observed_workload(text):
    command = re.search(r'^Command: (.+)$', text, re.MULTILINE)
    runtime = re.search(r'non-default args: (\{.*\})', text)
    if not command or not runtime:
        raise RuntimeError('Missing benchmark command or runtime arguments')
    actual = ast.literal_eval(runtime[1])
    expected = {'seed': 1024, 'dtype': 'bfloat16', 'block_size': 256,
                'max_num_batched_tokens': 4096, 'gpu_memory_utilization': 0.95,
                'enforce_eager': True}
    if any(actual.get(key) != value for key, value in expected.items()):
        raise IntegrityFailure('Runtime workload differs from the frozen R4 protocol')
    options = {}
    for flag in ('model', 'dataset', 'num-prompts', 'output-len', 'seed',
                 'dtype', 'block-size', 'max-num-batched-tokens', 'gpu-memory-utilization'):
        value = re.search(r'--' + flag + r'\s+(\S+)', command[1])
        if not value:
            raise RuntimeError('Missing benchmark option: ' + flag)
        options[flag.replace('-', '_')] = value[1]
    if options['num_prompts'] != '32' or options['output_len'] != '512':
        raise IntegrityFailure('Unexpected R4 prompt count or output length')
    tokens = re.search(r'Total num prompt tokens:\s*(\d+)', text)
    output = re.search(r'Total num output tokens:\s*(\d+)', text)
    if not tokens or not output or int(tokens[1]) != 6854 or int(output[1]) != 16384:
        raise RuntimeError('Incomplete or changed R4 benchmark token count')
    return {**options, **expected, 'command': command[1],
            'input_tokens': int(tokens[1]), 'output_tokens': int(output[1])}


class HardwareEvaluator:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / 'strategy_search.py').write_bytes(Path(__file__).read_bytes())
        self.policy = args.kernel_root / 'dlc_src/opt_flag_data/autotune_strategies.csv'
        self.library = args.library.resolve()
        self.original = args.original_policy.read_bytes()
        self.original_library = args.original_library.read_bytes()
        (self.out / 'original.csv').write_bytes(self.original)
        (self.out / 'original.so').write_bytes(self.original_library)
        self.expected = digest(self.policy.read_bytes())
        self.current_library = digest(self.library.read_bytes())
        self.sources = self.source_identity()
        self.cache, self.counter, self.workload = {}, 0, None
        self.environment = dict(os.environ)
        self.environment.update({
            # setproctitle otherwise overwrites the environment memory read via /proc.
            # Preserve it before spawning so the worker checks below remain valid.
            'SPT_NOENV': '1',
            'DLC_VISIBLE_DEVICES': args.device, 'CHIPLTECH_VISIBLE_DEVICES': args.device,
            'VLLM_TARGET_DEVICE': 'dlc', 'VLLM_PROFILE_MAX_NUM_BATCHED_TOKENS': '4096',
            'VLLM_PROFILE_NUM_PROMPTS': '32', 'VLLM_PROFILE_OUTPUT_LEN': '512',
            'VLLM_PROFILE_DTYPE': 'bfloat16', 'VLLM_PROFILE_BLOCK_SIZE': '256',
            'VLLM_PROFILE_TEMPERATURE': '0.0', 'VLLM_USE_DLC_COL_MAJOR_MATMUL': '1',
            'VLLM_PROFILE_DATASET': '/mnt/jfs/dataset/ShareGPT_V3_unfiltered_cleaned_split.json',
            'PYTHONUNBUFFERED': '1',
        })
        self.environment.update({key: '1' if args.profiling == 'on' else '0'
                                 for key in ('DLC_SYN_DEBUG', 'DLC_SYN_VERBOSE', 'DLC_SYN_PROF_CYCLE')})

    def source_identity(self):
        result = {}
        for name, root, excludes in (
            ('kernel', self.args.kernel_root, [':!dlc_src/opt_flag_data/autotune_strategies.csv']),
            ('tuner', self.args.tuner_root, []),
        ):
            result[name] = {
                'head': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
                'diff_sha': digest(subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD', '--binary', '--', '.', *excludes])),
            }
            untracked = subprocess.check_output(
                ['git', '-C', str(root), 'ls-files', '--others', '--exclude-standard', '-z']).decode().split('\0')
            result[name]['untracked_sources'] = {
                path: digest((root / path).read_bytes()) for path in sorted(untracked)
                if path and ((root / path).suffix in ('.py', '.c', '.cc', '.cpp', '.h', '.hpp', '.cmake')
                             or (root / path).name == 'CMakeLists.txt')}
        result['runner_sha'] = digest((self.args.tuner_root / 'DLC/vllm_profile_runner.py').read_bytes())
        result['algorithm_sha'] = digest(Path(__file__).read_bytes())
        result['model_config_sha'] = digest(self.args.model_config.read_bytes())
        return result

    def status(self, **fields):
        atomic_json(self.out / 'status.json', {'pid': os.getpid(), 'time': time.time(), **fields})
        print(json.dumps(fields), flush=True)

    def assert_integrity(self):
        if (digest(self.policy.read_bytes()) != self.expected
                or digest(self.library.read_bytes()) != self.current_library
                or self.source_identity() != self.sources):
            (self.out / 'unexpected-policy.csv').write_bytes(self.policy.read_bytes())
            atomic_json(self.out / 'unexpected-identity.json', {
                'policy_sha': digest(self.policy.read_bytes()),
                'library_sha': digest(self.library.read_bytes()),
                'sources': self.source_identity()})
            raise IntegrityFailure('Source, policy or library changed outside this run; evidence preserved')

    def command(self, command, log, timeout, observe=None):
        """Bound a candidate and reap only its owned runner/worker group."""
        with log.open('w') as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                       env=self.environment, start_new_session=True)
            try:
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    try:
                        return process.wait(timeout=min(1, max(.01, deadline - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        if observe:
                            observe(process.pid)
                output.write(f'Command timed out after {timeout} seconds\n')
                return 124
            finally:
                def live_group():
                    process.poll()
                    groups = subprocess.check_output(['ps', '-eo', 'pgid=,stat='], text=True)
                    return any(parts[0] == str(process.pid) and not parts[1].startswith('Z')
                               for parts in (line.split() for line in groups.splitlines()) if len(parts) == 2)
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    if live_group():
                        try:
                            os.killpg(process.pid, sig)
                        except ProcessLookupError:
                            pass
                    deadline = time.monotonic() + 5
                    while live_group() and time.monotonic() < deadline:
                        time.sleep(.05)
                if live_group():
                    self.cleanup_failed = True
                    raise CleanupFailure('Live workers remain in owned group ' + str(process.pid))
                process.wait(timeout=5)

    def install(self, data, directory):
        self.assert_integrity()
        atomic_bytes(self.policy, data)
        self.expected = digest(data)
        key = policy_identity(data)
        if key not in self.cache:
            root = str(self.args.kernel_root)
            # Keep the reusable library, not a second full install tree per policy.
            with tempfile.TemporaryDirectory(prefix='install-', dir=directory) as staging:
                for name, command in (
                    ('cmake', ['cmake', '-S', root, '-B', root + '/build', '-DCMAKE_BUILD_TYPE=Release']),
                    ('build', ['ninja', '-C', root + '/build', '-j' + str(self.args.build_jobs)]),
                    ('stage', ['cmake', '--install', root + '/build', '--prefix', staging]),
                ):
                    if self.command(command, directory / (name + '.log'), 900):
                        raise RuntimeError('Candidate build failed: ' + name)
                self.assert_integrity()
                compiled = (Path(staging) / 'lib/libcustom_dlc_perf_lib.so').read_bytes()
                (directory / 'library.so').write_bytes(compiled)
                self.cache[key] = directory / 'library.so'
        self.assert_integrity()
        atomic_bytes(self.library, self.cache[key].read_bytes())
        self.current_library = digest(self.library.read_bytes())

    def benchmark_command(self, directory):
        profile = directory / 'profile'
        profile.mkdir(exist_ok=True)
        self.environment['DLC_SYN_LOG_DIR'] = str(profile)
        enabled = '1' if self.args.profiling == 'on' else '0'
        self.environment.update({k: enabled for k in ('DLC_SYN_DEBUG', 'DLC_SYN_VERBOSE', 'DLC_SYN_PROF_CYCLE')})
        return ['python3', '-u', str(self.args.tuner_root / 'DLC/vllm_profile_runner.py'),
                '--model-name', 'Qwen3-8B', '--config', str(self.args.model_config),
                '--model-path', str(self.args.model), '--device', self.args.device,
                '--profile-dir', str(profile), '--profiling', self.args.profiling]

    def run(self, data, label):
        self.counter += 1
        directory = self.out / f'{self.counter:03d}-{label}'
        directory.mkdir()
        (directory / 'strategy.csv').write_bytes(data)
        started = time.monotonic()
        result = {'label': label, 'exit_code': None, 'strategy_sha': digest(data),
                  'path': str(directory), 'profiling': self.args.profiling,
                  'completed_32': False, 'recovery_epoch': 0}
        error = None
        try:
            self.status(phase='compile', run=self.counter, label=label, strategy_sha=digest(data))
            self.install(data, directory)
            info, loaded = self.library.stat(), {}
            log = directory / 'benchmark.log'

            def observe(group):
                if loaded or not log.exists():
                    return
                for pid in set(re.findall(r'EngineCore pid=(\d+)', log.read_text(errors='replace'))):
                    process = Path('/proc') / pid
                    try:
                        pgid = int((process / 'stat').read_text().rpartition(')')[2].split()[2])
                        maps = (process / 'maps').read_text().splitlines()
                        if pgid != group:
                            continue
                        for line in maps:
                            parts = line.split()
                            if len(parts) < 6 or parts[5] != str(self.library):
                                continue
                            major, minor = (int(v, 16) for v in parts[3].split(':'))
                            if int(parts[4]) != info.st_ino or os.makedev(major, minor) != info.st_dev:
                                raise IntegrityFailure('Worker mapped a different library identity')
                            env = dict(item.split(b'=', 1) for item in
                                       (process / 'environ').read_bytes().split(b'\0') if b'=' in item)
                            keys = ('DLC_SYN_DEBUG', 'DLC_SYN_VERBOSE', 'DLC_SYN_PROF_CYCLE',
                                    'DLC_VISIBLE_DEVICES', 'VLLM_USE_DLC_COL_MAJOR_MATMUL')
                            flags = {k: env.get(k.encode(), b'').decode() for k in keys}
                            expected_flags = {k: self.environment[k] for k in keys}
                            if flags != expected_flags:
                                result['worker_environment'] = {
                                    'pid': int(pid), 'expected': expected_flags, 'actual': flags,
                                }
                                raise IntegrityFailure('Worker device, instrumentation or layout differs')
                            loaded.update(pid=int(pid), pgid=pgid, inode=info.st_ino,
                                          device=info.st_dev, path=str(self.library), flags=flags,
                                          sha=self.current_library)
                    except (FileNotFoundError, ProcessLookupError):
                        continue

            command = self.benchmark_command(directory)
            self.status(phase='benchmark', run=self.counter, label=label,
                        strategy_sha=self.expected, library_sha=self.current_library)
            result['library_sha'] = self.current_library
            result['argv'] = command
            result['exit_code'] = self.command(command, log, 600, observe=observe)
            text = log.read_text(errors='replace')
            match = re.search(r'Throughput:\s*[\d.]+ requests/s,\s*([\d.]+) total tokens/s,\s*([\d.]+) output tokens/s', text)
            result['completed_32'] = bool(re.search(r'Processed prompts:.*100%.*32/32', text))
            if result['exit_code'] != 0 or not match or not result['completed_32']:
                raise RuntimeError('Incomplete benchmark')
            actual = observed_workload(text)
            if self.workload is not None and actual != self.workload:
                raise IntegrityFailure('Workload changed between strategies')
            self.assert_integrity()
            if not loaded:
                raise IntegrityFailure('Missing EngineCore mapped-library receipt')
            total, output = float(match[1]), float(match[2])
            if not all(math.isfinite(v) and v > 0 for v in (total, output)):
                raise RuntimeError('Invalid throughput')
            self.workload = actual
            result.update(total=total, output=output, workload=actual, loaded_library=loaded)
        except (RuntimeError, ValueError, IntegrityFailure, CleanupFailure) as failure:
            result['error'] = str(failure)
            error = failure
        result['seconds'] = time.monotonic() - started
        atomic_json(directory / 'measurement.json', result)
        with (self.out / 'measurements.jsonl').open('a') as log:
            log.write(json.dumps(result, allow_nan=False) + '\n')
        self.status(phase='measured', **result)
        if error:
            if isinstance(error, (IntegrityFailure, CleanupFailure)):
                raise error
            raise RuntimeError('Failed measurement: ' + str(directory) + ': ' + str(error)) from error
        result['measurement_sha'] = digest((directory / 'measurement.json').read_bytes())
        return result

    def retry_run(self, data, label):
        for attempt in range(3):
            try:
                return self.run(data, f'{label}-attempt-{attempt+1}')
            except RuntimeError as error:
                self.status(phase='retry', label=label, attempt=attempt+1, error=str(error))
                if attempt == 2:
                    raise

    def restore(self):
        if getattr(self, 'cleanup_failed', False):
            raise CleanupFailure('Owned live workers remain; original policy/library preserved in artifacts')
        self.assert_integrity()
        atomic_bytes(self.policy, self.original)
        atomic_bytes(self.library, self.original_library)
        self.expected, self.current_library = digest(self.original), digest(self.original_library)



def calibrate_search(evaluator, base, donors, *, budget=None):
    """Keep profiling hints separate from scores in the requested runtime mode."""
    profiling = evaluator.args.profiling
    hints = None
    limit = len(donors) if budget is None else min(len(donors), budget-1)

    def donor_runs(prefix):
        results = []
        for index, data in enumerate(donors):
            result = None
            if index < limit:
                try:
                    result = evaluator.retry_run(data, f'{prefix}-{index+1}')
                except RuntimeError as error:
                    evaluator.status(phase='donor-calibration-failed', donor_sha=digest(data), error=str(error))
            results.append(result)
        return results

    if profiling == 'off':
        flags = ('DLC_SYN_DEBUG', 'DLC_SYN_VERBOSE', 'DLC_SYN_PROF_CYCLE')
        previous = {key: evaluator.environment[key] for key in flags}
        try:
            evaluator.args.profiling = 'on'
            evaluator.environment.update({key: '1' for key in flags})
            hints = [evaluator.retry_run(base, 'hint-profile-base'), *donor_runs('hint-profile-donor')]
            atomic_json(evaluator.out / 'hint-calibration.json', {'runs': hints, 'scoring': False})
        finally:
            evaluator.args.profiling = profiling
            evaluator.environment.update(previous)
    baselines = [evaluator.retry_run(base, f'calibrate-base-{i+1}') for i in range(3)]
    baseline_total = statistics.median(r['total'] for r in baselines)
    atomic_json(evaluator.out / 'baseline-calibration.json', {
        'runs': baselines, 'median_total': baseline_total, 'profiling': profiling,
        'mean_total': statistics.mean(r['total'] for r in baselines)})
    measured = {policy_identity(base): baseline_total}
    if hints is None:
        alternates = donor_runs('calibrate-donor')
        hints = [baselines[-1], *alternates]
        measured.update({policy_identity(data): result['total'] if result else None
                         for data, result in zip(donors[:limit], alternates[:limit])})
    profiles = [summarize_profiles(Path(result['path']).glob('profile/syn_*.ansi')) if result else {}
                for result in hints]
    guidance = []
    for data, profile in zip(donors, profiles[1:]):
        gains, recipes = cycle_guidance(base, data, profiles[0], profile)
        guidance.append({'donor_sha': digest(data), 'gains': gains, 'families': list(recipes),
                         'donor_profile': profile})
    atomic_json(evaluator.out / 'guidance.json', {
        'base_profile': profiles[0], 'donors': guidance, 'scoring_profiling': profiling})
    return profiles[0], profiles[1:], measured


def run_search(evaluator, args):
    manifest = json.loads((args.inputs / 'manifest.json').read_text())
    base = (args.inputs / 'base.csv').read_bytes()
    reference = (args.inputs / 'reference.csv').read_bytes()
    donors = [(args.inputs / f'donor-{i+1}.csv').read_bytes()
              for i in range(len(manifest['donor_shas']))]
    if (digest(base) != manifest['base_sha'] or digest(reference) != manifest['reference_sha']
            or [digest(data) for data in donors] != manifest['donor_shas']):
        raise IntegrityFailure('Prepared search inputs changed')
    if policy_identity(reference) in {policy_identity(data) for data in [base, *donors]}:
        raise IntegrityFailure('Independent control was included in search seeds')
    (evaluator.out / 'reference.csv').write_bytes(reference)
    base_stats, alt_stats, measured = calibrate_search(evaluator, base, donors, budget=args.budget)

    def measure(data):
        key = policy_identity(data)
        if key not in measured:
            try:
                measured[key] = evaluator.run(data, 'search-' + key[:10])['total']
            except RuntimeError as error:
                evaluator.status(phase='candidate-failed', error=str(error))
                measured[key] = None
        return measured[key]

    def on_round(state):
        atomic_json(evaluator.out / 'search-state.json', state)
        evaluator.status(phase='search-round', depth=state['round'], measured=len(measured))

    def confirm(data):
        try:
            return evaluator.retry_run(data, 'finalist-' + policy_identity(data)[:10])['total']
        except RuntimeError as error:
            evaluator.status(phase='finalist-failed', error=str(error))
            return None

    selected = search_policy(base, donors, base_stats, alt_stats, measure,
                             budget=args.budget, rounds=args.rounds,
                             on_round=on_round, confirm=confirm)
    if len(selected['finalists']) < 1 or any(f['median'] is None for f in selected['finalists']):
        raise RuntimeError('R4 finalist pool did not complete all repeat measurements')
    candidate = selected.pop('strategy')
    (evaluator.out / 'candidate.csv').write_bytes(candidate)
    atomic_json(evaluator.out / 'selected-search.json', selected)
    return reference, candidate


def run_stage(args):
    """Search/off and fixed-winner/on are separate workflow steps and receipts."""
    primary = None
    if args.phase == 'qualify':
        primary = json.loads((args.source / 'report.json').read_text())
        if not primary['passed']:
            args.output.mkdir(parents=True, exist_ok=False)
            atomic_json(args.output / 'report.json', {
                'passed': False, 'profiling': 'on', 'skipped': 'Primary independent A/B did not pass'})
            atomic_json(args.output.parent / 'acceptance.json', {'passed': False})
            return
        previous = json.loads((args.source / 'environment.json').read_text())
        if (previous['device'] != args.device
                or digest(args.original_policy.read_bytes()) != previous['original_strategy_sha']
                or digest(args.original_library.read_bytes()) != previous['original_library_sha']
                or digest((args.source / 'original.csv').read_bytes()) != previous['original_strategy_sha']
                or digest((args.source / 'original.so').read_bytes()) != previous['original_library_sha']):
            raise IntegrityFailure('Assigned device or original rollback snapshots changed after search')
        args = argparse.Namespace(**vars(args))
        args.original_policy, args.original_library = args.source / 'original.csv', args.source / 'original.so'
        cleanup = json.loads((args.source / 'cleanup.json').read_text())
        selected = json.loads((args.source / 'selected-search.json').read_text())
        candidate = (args.source / 'candidate.csv').read_bytes()
        reference = (args.source / 'reference.csv').read_bytes()
        policy = args.kernel_root / 'dlc_src/opt_flag_data/autotune_strategies.csv'
        if (not selected['auto_selected'] or not primary['automatic_search']
                or primary['profiling'] != 'off' or not cleanup['retained'] or cleanup['cleanup_failed']
                or digest(candidate) != selected['candidate_sha'] or digest(candidate) != primary['candidate_sha']
                or digest(reference) != primary['reference_sha']
                or digest(policy.read_bytes()) != cleanup['strategy_sha']
                or digest(args.library.read_bytes()) != cleanup['library_sha']):
            raise IntegrityFailure('Qualification requires the unchanged primary automatic winner and cleanup')
    else:
        manifest = json.loads((args.inputs / 'manifest.json').read_text())
        if (digest(args.original_policy.read_bytes()) != manifest['original_strategy_sha']
                or digest(args.original_library.read_bytes()) != manifest['original_library_sha']):
            raise IntegrityFailure('Original rollback snapshots changed after input preparation')

    with open(f'/tmp/dlc-autotune-device-{args.device}.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluator = HardwareEvaluator(args)
        retained = False
        try:
            if primary:
                evaluator.sources = previous['sources']
                evaluator.assert_integrity()
                evaluator.workload = json.loads((args.source / 'baseline-calibration.json').read_text())['runs'][-1]['workload']
                (evaluator.out / 'reference.csv').write_bytes(reference)
                (evaluator.out / 'candidate.csv').write_bytes(candidate)
            atomic_json(evaluator.out / 'environment.json', {
                'sources': evaluator.sources, 'device': args.device, 'profiling': args.profiling,
                'budget': args.budget, 'rounds': args.rounds, 'width': 2, 'fanout': 6,
                'selection_source': str(args.source) if primary else None,
                'original_strategy_sha': digest(evaluator.original),
                'original_library_sha': digest(evaluator.original_library)})
            if not primary:
                reference, candidate = run_search(evaluator, args)
            pairs = confirm_pairs(evaluator.retry_run, reference, candidate,
                                  on_pair=lambda pairs: atomic_json(evaluator.out / 'paired-results.json', pairs))
            report = {**confirmation_report(pairs), 'automatic_search': True,
                      'candidate_sha': digest(candidate), 'reference_sha': digest(reference),
                      'profiling': args.profiling,
                      'sources': evaluator.sources, 'device': args.device,
                      'selection_source': str(args.source) if primary else str(evaluator.out)}
            atomic_json(evaluator.out / 'report.json', report)
            if report['passed']:
                evaluator.install(candidate, evaluator.out)
                retained = True
            evaluator.status(phase='finished', **report)
        finally:
            try:
                if not retained:
                    evaluator.restore()
                evaluator.assert_integrity()
                cleanup = {'retained': retained, 'cleanup_failed': False,
                           'strategy_sha': digest(evaluator.policy.read_bytes()),
                           'library_sha': digest(evaluator.library.read_bytes())}
            except BaseException as error:
                cleanup = {'retained': False, 'cleanup_failed': True, 'error': str(error)}
                raise
            finally:
                atomic_json(evaluator.out / 'cleanup.json', cleanup)
                evaluator.status(phase='cleanup', **cleanup)
                if primary:
                    atomic_json(evaluator.out.parent / 'acceptance.json', {
                        'passed': retained and not cleanup['cleanup_failed'],
                        'candidate_sha': digest(candidate), 'reference_sha': digest(reference),
                        'primary_report': str(args.source / 'report.json'),
                        'qualification_report': str(evaluator.out / 'report.json'),
                        'primary_report_sha': digest((args.source / 'report.json').read_bytes()),
                        'qualification_report_sha': digest((evaluator.out / 'report.json').read_bytes())
                        if (evaluator.out / 'report.json').exists() else None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'search', 'qualify'))
    parser.add_argument('--seed-dir', type=Path)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--original-policy', type=Path, required=True)
    parser.add_argument('--original-library', type=Path)
    parser.add_argument('--kernel-root', type=Path)
    parser.add_argument('--tuner-root', type=Path)
    parser.add_argument('--model-config', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library', type=Path,
                        default=Path('/usr/local/chipltech/synapse/lib/libcustom_dlc_perf_lib.so'))
    parser.add_argument('--device', default=os.environ.get('CHIPLTECH_VISIBLE_DEVICES', os.environ.get('DLC_VISIBLE_DEVICES', '0')))
    parser.add_argument('--build-jobs', type=int, default=14)
    parser.add_argument('--budget', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=9)
    args = parser.parse_args()
    if args.phase == 'prepare':
        if not args.seed_dir or not args.original_library:
            parser.error('prepare requires seed-dir and original-library')
        prepare_inputs(args)
        return
    if not re.fullmatch(r'\d+', args.device) or args.budget < 2 or args.rounds < 1 or args.build_jobs < 1:
        parser.error('Use one assigned device and positive search/build limits')
    required = ('original_library', 'kernel_root', 'tuner_root', 'model_config', 'model',
                'inputs' if args.phase == 'search' else 'source')
    if any(getattr(args, name) is None for name in required):
        parser.error('Missing stage arguments: ' + ', '.join(required))
    args.profiling = 'off' if args.phase == 'search' else 'on'
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    run_stage(args)


if __name__ == '__main__':
    main()
