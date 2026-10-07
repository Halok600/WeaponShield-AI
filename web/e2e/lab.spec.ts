import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { expect, type Page, test } from '@playwright/test';

const bus = fileURLToPath(new URL('../../fixtures/golden/images/bus.jpg', import.meta.url));

const busGolden = JSON.parse(readFileSync(fileURLToPath(new URL('../../fixtures/golden/bus.json', import.meta.url)), 'utf8')) as {
  ultralytics: { classId: number; conf: number; box: [number, number, number, number] }[];
};

type Box = [number, number, number, number];
interface Det {
  cls: string;
  conf: number;
  box: Box;
}

function iou(a: Box, b: Box): number {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
  const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
}

async function readDetections(page: Page): Promise<Det[]> {
  await expect.poll(async () => (await page.getByTestId('detections-json').textContent()) ?? '', { timeout: 30_000 }).not.toBe('');
  return JSON.parse((await page.getByTestId('detections-json').textContent()) ?? '[]') as Det[];
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('status')).toHaveText('ready', { timeout: 60_000 });
});

test('page is cross-origin isolated so WASM threads are available', async ({ page }) => {
  expect(await page.evaluate(() => crossOriginIsolated)).toBe(true);
});

test('detects people in an uploaded image', async ({ page }) => {
  await expect(page.getByTestId('backend')).toHaveText(/webgpu|wasm/);
  await page.getByTestId('image-input').setInputFiles(bus);
  await expect(page.getByTestId('count-person')).not.toHaveText('0', { timeout: 30_000 });
  const people = Number(await page.getByTestId('count-person').textContent());
  expect(people).toBeGreaterThanOrEqual(3);
});

test('processes webcam frames continuously', async ({ page }) => {
  await page.getByTestId('webcam-toggle').click();
  await expect.poll(async () => Number(await page.getByTestId('frames').textContent()), { timeout: 30_000 }).toBeGreaterThan(5);
  await page.getByTestId('webcam-toggle').click();
});

// The worker letterboxes with its own drawImage; the Node golden test feeds Ultralytics' pre-letterboxed pixels,
// so this is the only check that the browser preprocessing matches the reference.
test('browser letterbox matches Ultralytics on the golden bus image', async ({ page }) => {
  await page.getByTestId('image-input').setInputFiles(bus);
  const ours = await readDetections(page);
  const people = busGolden.ultralytics.filter((d) => d.classId === 0 && d.conf >= 0.5);
  expect(people.length).toBeGreaterThanOrEqual(3);
  for (const ref of people) {
    const best = Math.max(0, ...ours.filter((d) => d.cls === 'person').map((d) => iou(d.box, ref.box)));
    expect(best, `no person with IoU >= 0.9 for reference box [${ref.box.join(', ')}]`).toBeGreaterThanOrEqual(0.9);
  }
});

test('falls back to WASM when WebGPU is disabled and still detects', async ({ page }) => {
  await page.goto('/?webgpu=0');
  await expect(page.getByTestId('status')).toHaveText('ready', { timeout: 60_000 });
  await expect(page.getByTestId('backend')).toHaveText('wasm');
  await expect(page.getByTestId('adapter')).toContainText('none');
  await page.getByTestId('image-input').setInputFiles(bus);
  const people = (await readDetections(page)).filter((d) => d.cls === 'person');
  expect(people.length).toBeGreaterThanOrEqual(3);
});
