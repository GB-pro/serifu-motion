import ort from 'onnxruntime-node';
for (const f of process.argv.slice(2)) { const s = await ort.InferenceSession.create(f); console.log(f.split('/').pop()); console.log(' in', s.inputNames, ' out', s.outputNames); try { console.log(JSON.stringify(s.inputMetadata).slice(0, 600)); } catch (e) {} }
