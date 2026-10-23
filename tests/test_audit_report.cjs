const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const file = path.join(root, '实验文档/11-输入审计报告.html');
const html = fs.readFileSync(file, 'utf8');
const data = JSON.parse(html.match(/<script id="audit-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const script = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)][0][1];
const context = vm.createContext({document: {querySelector: () => ({textContent: JSON.stringify(data)})}});
vm.runInContext(script.split("$('#search').addEventListener")[0], context);
const check = value => {context.input = value; return vm.runInContext('validateSnapshot(input); buildModel(input)', context);};
const model = check(data);
assert.equal(model.rows.length, 26);
assert.equal(model.finished, data.groups.reduce((n, g) => n + g.reports.length, 0));
assert.equal(model.full, true);

const missing = structuredClone(data);
delete missing.groups.find(g => g.reports.length).reports[0].totals;
assert.throws(() => check(missing));

const final = structuredClone(data);
const template = final.groups.find(g => g.reports.length).reports[0];
const names = [...final.splits.train, ...final.splits.validation];
final.groups = [{protocol: {databases: names}, reports: names.map(database => ({...structuredClone(template), database}))}];
final.run_status = 'completed time=2026-10-02T16:00:00+08:00';
final.summary = {expected_databases: names, missing_databases: [], duplicate_databases: [], failed_databases: [], checked_rows: 1300000};
assert.equal(check(final).full, true);

const shifted = structuredClone(final);
shifted.groups[0].reports[0].totals.checked_rows = 45000;
shifted.groups[0].reports[1].totals.checked_rows = 55000;
assert.throws(() => check(shifted));

const incomplete = structuredClone(final);
incomplete.groups[0].reports[0].totals.recreated_rows = 0;
assert.throws(() => check(incomplete));

const unknown = structuredClone(final);
unknown.groups[0].reports[0] = {database: names[0], alignment_passed: false, fatal_error: 'fixture failure'};
assert.equal(check(unknown).full, false);
assert.equal(check(unknown).failed, 1);

const summaryMissing = structuredClone(final);
summaryMissing.summary.missing_databases = [names[0]];
assert.equal(check(summaryMissing).full, false);

assert.ok(!/<script[^>]+src=/i.test(html));
assert.ok(html.includes('重现') || html.includes('重算'));
console.log('Audit HTML checks passed: live progress, strict per-database completion, invalid imports and offline assets.');
