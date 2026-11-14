// node tools/dev/understand_test.mjs : the understanding module against imitated answers of the external AI (no key needed)
import assert from 'assert';
import { understandRemote, understandLocal, toOverrides, buildRequest, parseJson, tidy, API_URL } from '../../app/core/understand.js';
const lines = [
  { speaker: 'A', start: 0, end: 4, text: '今日は本当に楽しかった！ありがとう。', lang: 'ja' },
  { speaker: 'B', start: 4.5, end: 9, text: 'でも、もう会えないんだね。', lang: 'ja' },
  { speaker: 'A', start: 9.5, end: 12, text: '明日の会議は十時からです。', lang: 'ja' },
];
const answer = { lines: [{ i: 0, parts: [{ share: 0.6, emo: 'HAP', intensity: 0.9, emphasis: '楽しかった' }, { share: 0.4, emo: 'HAP', intensity: 0.6, emo2: 'SAD' }] }, { i: 1, parts: [{ share: 2, emo: 'SAD', intensity: 5 }] }, { i: 2, parts: [{ share: 1, emo: 'XXX', intensity: 1 }] }, { i: 9, parts: [] }] };
const reply = (body, status = 200) => async (url, init) => { reply.seen = { url, init }; return { ok: status < 300, status, json: async () => body }; };
// 1. a good answer, wrapped in a code fence and preceded by a thinking block
let out = await understandRemote(lines, { key: 'test', fetchFn: reply({ stop_reason: 'end_turn', content: [{ type: 'thinking', thinking: '' }, { type: 'text', text: '```json\n' + JSON.stringify(answer) + '\n```' }] }) });
assert.equal(reply.seen.url, API_URL); const h = reply.seen.init.headers, sent = JSON.parse(reply.seen.init.body);
assert.equal(h['x-api-key'], 'test'); assert.ok(h['anthropic-version']); assert.equal(h['anthropic-dangerous-direct-browser-access'], 'true');
assert.equal(sent.messages.length, 1); assert.equal(JSON.parse(sent.messages[0].content).lines.length, 3);
assert.equal(out[0].length, 2); assert.equal(out[1][0].intensity, 1.3); assert.equal(out[1][0].share, 1); assert.equal(out[2], null);
const ov = toOverrides(lines.filter((l) => l.speaker === 'A'), [out[0], out[2]], [{ start: 3, end: 3.5, emo: 'ANG', intensity: 1 }, { start: 0, end: 1, emo: 'NEU', intensity: 0.5, ai: true }]);
assert.deepEqual(ov.map((o) => [o.start, o.end, o.emo, !!o.ai]), [[0, 2.4, 'HAP', true], [3, 3.5, 'ANG', false]]); // the hand-set mark wins over the part it overlaps, the old AI mark is gone
// 2. failures carry a reason
for (const [f, code] of [[reply({ error: { message: 'invalid x-api-key' } }, 401), 'key'], [reply({}, 429), 'rate'], [reply({ stop_reason: 'refusal', content: [] }), 'refused'], [reply({ stop_reason: 'max_tokens', content: [{ type: 'text', text: '{"lines":[{"i":0,' }] }), 'answer'], [async () => { throw new TypeError('Failed to fetch'); }, 'network']])
  await assert.rejects(understandRemote(lines, { key: 'k', fetchFn: f }), (e) => e.code === code);
await assert.rejects(understandRemote(lines, {}), (e) => e.code === 'key');
// 3. the reader on the device
const loc = understandLocal([...lines, { text: "I'm not happy about this.", lang: 'en' }, { text: 'Oh no, watch out!', lang: 'en' }, { text: '全然楽しくない。', lang: 'ja' }]);
assert.equal(loc[0][0].emo, 'HAP'); assert.equal(loc[1][0].emo, 'SAD'); assert.equal(loc[2], null); assert.equal(loc[3][0].emo, 'SAD'); assert.equal(loc[4][0].emo, 'FEA'); assert.equal(loc[5][0].emo, 'SAD');
assert.equal(parseJson('x {"a":"}"} y').a, '}'); assert.equal(tidy(null, 2).length, 2); assert.ok(buildRequest(lines).system.length > 100);
console.log('understand: all checks passed');
