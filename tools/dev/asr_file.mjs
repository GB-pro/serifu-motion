// node tools/dev/asr_file.mjs <model dir> <tiny|base> <wav 16k mono> [lang] : the lines of a recording as the app cuts them, and what was said in each
import fs from 'fs'; import ort from 'onnxruntime-node';
import { Transcriber, parseTokens } from '../../app/core/asr.js';
import { extractRaw, speakerStats, normalize, FPS, SR } from '../../app/core/features.js';
import { analyzeSpeech } from '../../app/core/animate.js';
import { linesOf } from '../../app/core/reader.js';
import { readWav } from '../../train/wav.mjs';
const [dir, m, wav, lang] = process.argv.slice(2), o = { intraOpNumThreads: 2 };
const tr = new Transcriber(ort, await ort.InferenceSession.create(`${dir}/${m}-encoder.int8.onnx`, o), await ort.InferenceSession.create(`${dir}/${m}-decoder.int8.onnx`, o), parseTokens(fs.readFileSync(`${dir}/${m}-tokens.txt`, 'utf8')));
const x = readWav(wav), r = extractRaw(x), st = speakerStats([r]), F = normalize(r.raw, r.nFrames, st), lines = linesOf(analyzeSpeech(F, r.nFrames).phrases);
console.log('lines', lines.length, JSON.stringify(lines[0]).slice(0, 200));
for (const l of lines) { const t0 = Date.now(), a = l.t0 ?? l.f0, b = l.t1 ?? l.f1, s = x.subarray(Math.max(0, Math.round((a / FPS - 0.15) * SR)), Math.round((b / FPS + 0.15) * SR)); const out = await tr.line(s, { lang }); console.log((a / FPS).toFixed(2), (b / FPS).toFixed(2), out.lang, out.conf.toFixed(2), out.text, `(${Date.now() - t0} ms)`); }
