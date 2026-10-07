/** RGBA bytes (row-major, size x size) -> planar RGB float32 in [0,1] (NCHW without the batch dim). */
export function rgbaToTensor(rgba: ArrayLike<number>, size: number, out?: Float32Array): Float32Array {
  const pixels = size * size;
  if (rgba.length !== pixels * 4) {
    throw new Error(`rgbaToTensor: got ${rgba.length} bytes, expected ${pixels * 4}`);
  }
  const t = out ?? new Float32Array(pixels * 3);
  for (let i = 0, p = 0; i < pixels; i++, p += 4) {
    t[i] = rgba[p]! / 255;
    t[i + pixels] = rgba[p + 1]! / 255;
    t[i + 2 * pixels] = rgba[p + 2]! / 255;
  }
  return t;
}
