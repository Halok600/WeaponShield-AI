import { fileURLToPath } from 'node:url';
import { expect, test } from '@playwright/test';

const bus = fileURLToPath(new URL('../../fixtures/golden/images/bus.jpg', import.meta.url));

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
