"""The speech encoder of the reader: the first blocks of Whisper's encoder, cut out of a full encoder.

  python make_encoder.py <encoder.onnx of sherpa-onnx (k2-fsa) for a Whisper model> <out.onnx> <blocks>

Input 'mel' (1, 80, T): log-mel spectrogram (app/core/reader.js: logMel). Outputs 'h1' ... 'h<blocks>': the output of
every kept block, (1, T/2, width). Everything after the last kept block is removed, so the file holds only what is used.
Whisper (OpenAI) is under the MIT licence; the export to ONNX is from the sherpa-onnx project (Apache 2.0).
"""
import sys, onnx
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
outs = [f'/audioEncoder/blocks.{i}/Add_1_output_0' for i in range(n)]
onnx.utils.extract_model(src, dst, ['mel'], outs, check_model=False)
m = onnx.load(dst); g = m.graph
for i, o in enumerate(g.output):
    old, new = o.name, f'h{i + 1}'
    for node in g.node:
        for k, name in enumerate(node.output):
            if name == old: node.output[k] = new
        for k, name in enumerate(node.input):
            if name == old: node.input[k] = new
    o.name = new
del m.metadata_props[:]
for k, v in (('source', 'OpenAI Whisper encoder (MIT licence), ONNX export by sherpa-onnx; first %d blocks' % n),): p = m.metadata_props.add(); p.key, p.value = k, v
onnx.save(m, dst); onnx.checker.check_model(dst)
print(dst, 'nodes', len(g.node), 'outputs', [o.name for o in g.output], 'parameters', sum(int(__import__('numpy').prod(t.dims)) for t in g.initializer))
