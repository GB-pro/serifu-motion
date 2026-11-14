// node tools/dev/spk_emb_check.mjs <wav> <model> <ref_emb.txt> : the voiceprints speakers.js computes against reference ones (cosine near 1 = same features)
import fs from 'fs'; import ort from 'onnxruntime-node';
import { SpeakerSplitter } from '../../app/core/speakers.js'; import { readWav } from '../../train/wav.mjs';
const [wav, model, ref] = process.argv.slice(2), x = readWav(wav), sp = new SpeakerSplitter(ort, await ort.InferenceSession.create(model));
const R = fs.readFileSync(ref, 'utf8').trim().split('\n').map((l) => l.split(' ').map(Number));
const spans = JSON.parse(process.env.SPANS || "[[10,11.5],[30,31.5],[70,72]]");
for (let i = 0; i < spans.length; i++) { const e = await sp.embed(x.subarray(spans[i][0] * 16000, spans[i][1] * 16000)); let s = 0; for (let d = 0; d < R[i].length; d++) s += e[d] * R[i][d]; console.log(spans[i], 'cos', s.toFixed(4), 'dim', e.length); }
