// Downloads models listed in models.json (verified by SHA-256) and copies the
// ONNX Runtime Web wasm binaries into public/ so they are served same-origin.
import { createHash } from 'node:crypto';
import { copyFile, mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = join(dirname(fileURLToPath(import.meta.url)), '..');
const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

async function fetchModels() {
  const { models } = JSON.parse(await readFile(join(web, 'models.json'), 'utf8'));
  const dir = join(web, 'public', 'models');
  await mkdir(dir, { recursive: true });
  for (const m of models) {
    const dest = join(dir, m.file);
    try {
      if (sha256(await readFile(dest)) === m.sha256) continue;
    } catch {
      // missing file: download below
    }
    console.log(`prepare-assets: downloading ${m.file}`);
    const res = await fetch(m.url);
    if (!res.ok) throw new Error(`${m.url}: HTTP ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());
    const got = sha256(buf);
    if (got !== m.sha256) throw new Error(`${m.file}: sha256 ${got} does not match manifest ${m.sha256}`);
    await writeFile(dest, buf);
  }
}

async function copyOrtWasm() {
  const src = join(web, 'node_modules', 'onnxruntime-web', 'dist');
  const dest = join(web, 'public', 'ort');
  await mkdir(dest, { recursive: true });
  const files = (await readdir(src)).filter((f) => /^ort-wasm.*\.(wasm|mjs)$/.test(f));
  if (files.length === 0) throw new Error(`No ort-wasm files found in ${src}`);
  await Promise.all(files.map((f) => copyFile(join(src, f), join(dest, f))));
}

await Promise.all([fetchModels(), copyOrtWasm()]);
console.log('prepare-assets: ok');
