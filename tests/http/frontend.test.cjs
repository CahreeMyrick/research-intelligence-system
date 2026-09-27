const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const script = fs.readFileSync(path.resolve(__dirname, '../../../clarivue_qa/dist/app.js'), 'utf8');
function setup(fetch) {
  const elements = new Map();
  const element = selector => {
    if (!elements.has(selector)) elements.set(selector, {innerHTML:'', textContent:'', value:'', hidden:false, disabled:false, addEventListener(){}, setAttribute(){}, classList:{toggle(){},remove(){},add(){}}, focus(){},scrollIntoView(){}});
    return elements.get(selector);
  };
  const context = vm.createContext({document:{querySelector:element, querySelectorAll:()=>[], addEventListener(){}}, fetch, URL, AbortController, setTimeout, clearTimeout});
  vm.runInContext(script, context);
  return {element, context, run: code => vm.runInContext(code, context)};
}
const payload = {headline:'Result', paragraphs:['A claim [Evidence 1].'], citations:[{evidence_id:'evidence_1'}], sources:[{id:'evidence_1', title:'<script>bad</script>', passage:'Original text', summary:'Original text', url:'javascript:alert(1)'}], entities:[], graph_paths:[], missing_context:[], retrieval:{retrieved_count:1}};
test('built-in example submits to backend and renders real evidence/citations safely', async () => {
  const calls = [];
  const app = setup(async (...args) => {calls.push(args); return {ok:true,json:async()=>payload};});
  app.run("loadExample('running')");
  await app.run('submitQuestion({preventDefault(){}})');
  assert.equal(calls.length, 1);
  assert.equal(JSON.parse(calls[0][1].body).question, 'Why have I been developing chest pains while running?');
  assert.match(app.element('.source-grid').innerHTML, /id="source-evidence_1"/);
  assert.match(app.element('#answer').innerHTML, /data-source="evidence_1"/);
  assert.match(app.element('.source-grid').innerHTML, /&lt;script&gt;/);
  assert.doesNotMatch(app.element('.source-grid').innerHTML, /href="javascript:/);
});
test('backend errors leave no stale answer and unlock submission', async () => {
  const app = setup(async () => ({ok:false,json:async()=>({message:'Database unavailable'})}));
  app.element('#question').value = 'Test';
  await app.run('submitQuestion({preventDefault(){}})');
  assert.match(app.element('#notice').textContent, /Database unavailable/);
  assert.equal(app.element('#question-form button[type="submit"]').disabled, false);
  assert.equal(app.run('liveResponse'), null);
});
test('reset prevents late responses from replacing a new question', async () => {
  let resolve;
  const app = setup(() => new Promise(r => {resolve=r;}));
  app.element('#question').value = 'First';
  const pending = app.run('submitQuestion({preventDefault(){}})');
  app.run('clearRequest()');
  resolve({ok:true,json:async()=>payload});
  await pending;
  assert.equal(app.run('liveResponse'), null);
});
test('zero evidence has no undefined citation', async () => {
  const app = setup(async () => ({ok:true,json:async()=>({...payload,sources:[],citations:[],paragraphs:['Insufficient evidence.']})}));
  app.element('#question').value = 'Unknown';
  await app.run('submitQuestion({preventDefault(){}})');
  assert.doesNotMatch(app.element('#answer').innerHTML, /data-source/);
  assert.match(app.element('.source-grid').innerHTML, /No retrieved passages/);
});
