// node tools/dev/diar.mjs <wav 16k> <k> [model] [old] : speaker turns from speakers.js (or the old pitch-based diarize with "old") as "start end speaker" lines
import ort from 'onnxruntime-node'; import { readWav } from '../../train/wav.mjs';
import { SpeakerSplitter } from '../../app/core/speakers.js'; import { diarize } from '../../app/core/project.js';
const [wav, k = '2', model = '/root/data/spk/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx', mode] = process.argv.slice(2), x = readWav(wav);
const t0 = Date.now();
const segs = mode === 'old' ? diarize(x, +k).segments : (await new SpeakerSplitter(ort, await ort.InferenceSession.create(model, { intraOpNumThreads: 2 })).run(x, +k)).segments;
console.error('secs', ((Date.now() - t0) / 1000).toFixed(1));
for (const s of segs) console.log(s.start.toFixed(2), s.end.toFixed(2), s.speaker);
