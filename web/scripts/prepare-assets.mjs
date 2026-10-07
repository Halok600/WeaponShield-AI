// Downloads models listed in models.json (verified by SHA-256) and copies the
// ONNX Runtime Web wasm runtime into public/ort/<version>/ so it is served same-origin.
import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
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

// Only the asyncify variant is loaded by the 'onnxruntime-web/webgpu' bundle (verified by grepping the built
// worker); the jsep/jspi/plain variants would add ~60 MB of dead weight to every deploy. The files live under
// public/ort/<onnxruntime-web version>/ so a cache-first service worker can never serve a stale runtime
// after an upgrade.
const ORT_FILES = ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm'];

async function copyOrtWasm() {
  const pkg = join(web, 'node_modules', 'onnxruntime-web');
  const { version } = JSON.parse(await readFile(join(pkg, 'package.json'), 'utf8'));
  const root = join(web, 'public', 'ort');
  const dest = join(root, version);
  // Drop earlier layouts (unversioned files, other versions) so they do not ship.
  await rm(root, { recursive: true, force: true });
  await mkdir(dest, { recursive: true });
  await Promise.all(ORT_FILES.map((f) => copyFile(join(pkg, 'dist', f), join(dest, f))));
}

await Promise.all([fetchModels(), copyOrtWasm()]);
console.log('prepare-assets: ok');
