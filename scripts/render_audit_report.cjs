const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const folder = path.join(root, '实验文档/results/p1-input-audit-20261002');
const source = path.join(folder, 'server-results');
const read = name => JSON.parse(fs.readFileSync(path.join(source, name), 'utf8'));
const optional = name => fs.existsSync(path.join(source, name)) ? read(name) : null;
const capture = read('capture.json');
const data = JSON.parse(fs.readFileSync(path.join(root, 'configs/data.json'), 'utf8'));
const coverage = JSON.parse(fs.readFileSync(path.join(folder, 'coverage.json'), 'utf8'));
const groups = fs.readdirSync(source).filter(name => /^group-\d+$/.test(name)
    && fs.statSync(path.join(source, name)).isDirectory()).sort().map(group => ({
    group,
    protocol: read(`${group}/protocol.json`),
    progress: optional(`${group}/progress.json`),
    summary: optional(`${group}/summary.json`),
    reports: fs.readdirSync(path.join(source, group)).filter(name => name.endsWith('.json')
        && !['protocol.json', 'progress.json', 'summary.json'].includes(name))
        .sort().map(name => read(`${group}/${name}`)),
    log_excerpt: fs.readFileSync(path.join(source, `${group}.log`), 'utf8')
        .trim().split('\n').slice(-3),
}));
const snapshot = {
    schema: 1,
    audit: 'p1-input-audit-20261002',
    captured_at: capture.captured_at,
    run_status: fs.readFileSync(path.join(source, 'status.txt'), 'utf8').trim(),
    expected_rows: 1300000,
    splits: {train: data.splits.train, validation: data.splits.validation},
    groups,
    summary: optional('summary.json'),
    worker_processes: capture.worker_processes,
    test_evidence: {tests_passed: 65, audit_tests: coverage.tests,
        coverage_percent: coverage.coverage_percent, coverage_method: 'stdlib trace, not pytest-cov'},
};
const file = path.join(root, '实验文档/11-输入审计报告.html');
const template = fs.readFileSync(file, 'utf8');
const payload = JSON.stringify(snapshot).replace(/</g, '\\u003c');
const html = template.replace(/(<script id="audit-data" type="application\/json">)[\s\S]*?(<\/script>)/,
    (_, before, after) => before + payload + after);
if (html === template && !template.includes(payload)) throw Error('Embedded data block not found');
for (const script of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) new vm.Script(script[1]);
fs.writeFileSync(path.join(folder, 'report-snapshot.json'), JSON.stringify(snapshot, null, 2) + '\n');
fs.writeFileSync(file, html);
console.log(JSON.stringify({captured_at: snapshot.captured_at, status: snapshot.run_status,
    finished_databases: groups.reduce((n, g) => n + g.reports.length, 0), summary_ready: !!snapshot.summary}));
