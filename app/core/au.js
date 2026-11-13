// Serifu Motion — emo channel definitions (FACS Action Units + head/torso), shared with train/dataset.py
export const CHANNELS = [
  { id: 'AU1', group: 'brow', range: [0, 1], name: { en: 'Inner Brow Raiser', ja: '眉の内側を上げる' } },
  { id: 'AU2', group: 'brow', range: [0, 1], name: { en: 'Outer Brow Raiser', ja: '眉の外側を上げる' } },
  { id: 'AU4', group: 'brow', range: [0, 1], name: { en: 'Brow Lowerer', ja: '眉を下げ寄せる' } },
  { id: 'AU5', group: 'eye', range: [0, 1], name: { en: 'Upper Lid Raiser', ja: '上まぶたを上げる' } },
  { id: 'AU6', group: 'eye', range: [0, 1], name: { en: 'Cheek Raiser', ja: '頬を上げる' } },
  { id: 'AU7', group: 'eye', range: [0, 1], name: { en: 'Lid Tightener', ja: 'まぶたを緊張させる' } },
  { id: 'AU9', group: 'nose', range: [0, 1], name: { en: 'Nose Wrinkler', ja: '鼻にしわを寄せる' } },
  { id: 'AU10', group: 'mouth', range: [0, 1], name: { en: 'Upper Lip Raiser', ja: '上唇を上げる' } },
  { id: 'AU12', group: 'mouth', range: [0, 1], name: { en: 'Lip Corner Puller', ja: '口角を引き上げる' } },
  { id: 'AU14', group: 'mouth', range: [0, 1], name: { en: 'Dimpler', ja: 'えくぼを作る' } },
  { id: 'AU15', group: 'mouth', range: [0, 1], name: { en: 'Lip Corner Depressor', ja: '口角を下げる' } },
  { id: 'AU16', group: 'mouth', range: [0, 1], name: { en: 'Lower Lip Depressor', ja: '下唇を下げる' } },
  { id: 'AU17', group: 'mouth', range: [0, 1], name: { en: 'Chin Raiser', ja: '顎を上げる' } },
  { id: 'AU18', group: 'mouth', range: [0, 1], name: { en: 'Lip Pucker', ja: '唇をすぼめる' } },
  { id: 'AU20', group: 'mouth', range: [0, 1], name: { en: 'Lip Stretcher', ja: '唇を横に引く' } },
  { id: 'AU22', group: 'mouth', range: [0, 1], name: { en: 'Lip Funneler', ja: '唇を突き出す' } },
  { id: 'AU24', group: 'mouth', range: [0, 1], name: { en: 'Lip Pressor', ja: '唇を押し付ける' } },
  { id: 'AU26', group: 'mouth', range: [0, 1], name: { en: 'Jaw Drop', ja: '顎を下げる' } },
  { id: 'AU28', group: 'mouth', range: [0, 1], name: { en: 'Lip Suck', ja: '唇を吸い込む' } },
  { id: 'AU45', group: 'eye', range: [0, 1], name: { en: 'Blink', ja: '瞬き' } },
  { id: 'gazeX', group: 'gaze', range: [-1, 1], name: { en: 'Eyes Left/Right (AU61/62)', ja: '視線 左右 (AU61/62)' } },
  { id: 'gazeY', group: 'gaze', range: [-1, 1], name: { en: 'Eyes Up/Down (AU63/64)', ja: '視線 上下 (AU63/64)' } },
  { id: 'headPitch', group: 'head', range: [-25, 25], name: { en: 'Head Up/Down (AU53/54)', ja: '頭 上下 (AU53/54)' } },
  { id: 'headYaw', group: 'head', range: [-30, 30], name: { en: 'Head Turn (AU51/52)', ja: '頭 左右回転 (AU51/52)' } },
  { id: 'headRoll', group: 'head', range: [-20, 20], name: { en: 'Head Tilt (AU55/56)', ja: '頭 傾き (AU55/56)' } },
  { id: 'headZ', group: 'head', range: [-5, 5], name: { en: 'Head Forward/Back (AU57/58)', ja: '頭 前後 (AU57/58)' } },
  { id: 'torsoRoll', group: 'torso', range: [-10, 10], name: { en: 'Torso Tilt', ja: '胴体 傾き' } },
  { id: 'torsoShift', group: 'torso', range: [-6, 6], name: { en: 'Torso Sway', ja: '胴体 左右移動' } },
  { id: 'torsoShrug', group: 'torso', range: [-6, 6], name: { en: 'Shoulder Shrug', ja: '肩すくめ' } },
  { id: 'torsoLean', group: 'torso', range: [-6, 6], name: { en: 'Lean Forward/Back', ja: '前傾・後傾' } },
  // ---- acting channels: written by the animation layer (not predicted by the tracker-trained network) ----
  { id: 'AU43', group: 'eye', range: [0, 1], anim: true, name: { en: 'Lid Droop (AU43)', ja: 'まぶたを伏せる (AU43)' } },
  { id: 'browAsym', group: 'brow', range: [-1, 1], anim: true, name: { en: 'Brow L/R Offset', ja: '眉の左右差' } },
  { id: 'mouthAsym', group: 'mouth', range: [-1, 1], anim: true, name: { en: 'Mouth Corner L/R Offset', ja: '口角の左右差' } },
  { id: 'eyeScale', group: 'eye', range: [-1, 1], anim: true, name: { en: 'Eye Size', ja: '目の大きさ' } },
  { id: 'pupil', group: 'eye', range: [-1, 1], anim: true, name: { en: 'Pupil Size', ja: '瞳の大きさ' } },
  { id: 'squash', group: 'head', range: [-1, 1], anim: true, name: { en: 'Squash / Stretch', ja: 'つぶし・のばし' } },
  { id: 'headX', group: 'head', range: [-5, 5], anim: true, name: { en: 'Head Side Shift', ja: '頭 左右移動' } },
  { id: 'headY', group: 'head', range: [-5, 5], anim: true, name: { en: 'Head Up/Down Shift', ja: '頭 上下移動' } },
  { id: 'torsoYaw', group: 'torso', range: [-20, 20], anim: true, name: { en: 'Torso Twist', ja: '胴体 ひねり' } },
  { id: 'torsoY', group: 'torso', range: [-6, 6], anim: true, name: { en: 'Body Bounce', ja: '胴体 上下' } },
];
// the first KM channels are the network's outputs; the rest are acting channels
export const KM = 30;
export const IDS = CHANNELS.map((c) => c.id);
export const K = CHANNELS.length;
export const IDX = Object.fromEntries(IDS.map((id, i) => [id, i]));
export const EMOS = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS'];
export const GROUPS = ['brow', 'eye', 'nose', 'mouth', 'gaze', 'head', 'torso'];
// channels the camera may drive in live mode (blink + mouth shape only, per spec)
export const CAMERA_CHANNELS = ['AU45', 'AU26', 'AU22', 'AU18', 'AU20', 'AU16', 'AU10', 'AU24', 'AU28'];
export const MOUTH_CHANNELS = ['AU26', 'AU22', 'AU18', 'AU20', 'AU16', 'AU10', 'AU24', 'AU28', 'AU25'];

// MediaPipe/ARKit blendshape -> AU for the camera path (same mapping as training)
export function blendshapesToAU(bs) {
  const g = (n) => bs[n] || 0;
  const avg = (a, b) => (g(a) + g(b)) / 2;
  return {
    AU45: avg('eyeBlinkLeft', 'eyeBlinkRight'),
    AU26: g('jawOpen'), AU22: g('mouthFunnel'), AU18: g('mouthPucker'),
    AU20: avg('mouthStretchLeft', 'mouthStretchRight'), AU16: avg('mouthLowerDownLeft', 'mouthLowerDownRight'),
    AU10: avg('mouthUpperUpLeft', 'mouthUpperUpRight'), AU24: avg('mouthPressLeft', 'mouthPressRight'),
    AU28: avg('mouthRollLower', 'mouthRollUpper'),
  };
}
export function frameToObj(arr, off = 0) { const o = {}; for (let i = 0; i < K; i++) o[IDS[i]] = arr[off + i]; return o; }
export const N_AU = 20; // AU1..AU45 occupy the first 20 slots
