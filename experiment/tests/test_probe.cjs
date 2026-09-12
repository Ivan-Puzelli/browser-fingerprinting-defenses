// Execute the real inline probe with a mocked browser and collection endpoint.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../static/probe.html'), 'utf8');
const script = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)][0][1];
const tick = () => new Promise(resolve => setImmediate(resolve));

async function run(outcome, failProbe = false) {
  const elements = Object.fromEntries(['status', 'result', 'copy'].map(k =>
    [k, {textContent: '', addEventListener() {}}]));
  let state, resolvePost, rejectPost, posted;
  const context = {
    URLSearchParams, Date, performance: {now: () => 1},
    location: {search: '?browser=chrome&level=3&run=1', hostname: 'site-a.test', href: 'http://site-a.test'},
    navigator: {userAgent: 'test', clipboard: {writeText() {}}},
    document: {getElementById: id => elements[id], body: {setAttribute: (_, v) => { state = v; }}},
    fetch: (_, opts) => { posted = JSON.parse(opts.body); return new Promise((resolve, reject) => {
      resolvePost = resolve; rejectPost = reject;
    }); },
    FingerprintJS: {
      getRawCanvasFingerprint: () => ({textImage1: 'raw', textImage2: 'raw'}),
      getUnstableCanvasFingerprint: () => ({}),
      getDebugRendererInfoStatus: () => 'read-succeeded', wouldUpstreamSkipDebugRendererInfo: () => false,
      load: () => failProbe ? Promise.reject(new Error('probe failure')) : Promise.resolve({
        get: () => Promise.resolve({visitorId: 'id', components: {}})
      })
    }
  };
  vm.runInNewContext(script, context);
  await tick();
  assert.equal(state, undefined, 'must not finish before persistence confirmation');
  assert.equal(elements.status.textContent, 'saving…');
  if (outcome === 'network') rejectPost(new Error('offline'));
  else resolvePost({ok: outcome === 'ok', status: outcome === 'ok' ? 200 : 500});
  await tick();
  assert.equal(state, outcome === 'ok' && !failProbe ? 'done' : 'error');
  assert.equal(JSON.parse(elements.result.textContent).visitorId, posted.visitorId);
  if (failProbe) assert.equal(posted.error_stage, 'agent-get');
}

(async () => {
  await run('ok'); await run('network'); await run('http500');
  await run('ok', true); await run('network', true);
  console.log('PASS: pending save, confirmed save, HTTP error, network error, failed-probe persistence');
})().catch(err => { console.error(err); process.exitCode = 1; });
