# Plan 3 — Frontend (landing, console, model card, architecture)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Plan 1 lab page with the real product UI — a polished dark "security operations console" with a landing page that demos detection within seconds, a three-mode console (Live / Video / Image) with tracking overlays, alerts and an event timeline, an honest model card, and an architecture page — all running on the Plan 1 detection core.

**Architecture:** React Router data-mode SPA. One `DetectorProvider` owns the single `DetectorClient` (worker) for the app's lifetime and lazily loads the model. A `useVideoDetection` hook wraps `startLiveLoop` + `Pipeline`, paints each frame imperatively through a `Stage` handle (no per-frame React renders) and publishes throttled stats and an event log. UI primitives are shadcn-style components written in-repo on Radix primitives and Tailwind v4 design tokens. Charts use Recharts with a validated categorical palette.

**Tech Stack:** React 19, React Router 8, Tailwind CSS 4, Radix (tabs, toggle-group, tooltip), clsx + tailwind-merge, Lucide, Motion 14, Recharts 3, @fontsource-variable (Inter, JetBrains Mono), vite-plugin-pwa 2 (Workbox), Vitest 5 + Testing Library + jsdom, Playwright 1.63 + @axe-core/playwright.

**Spec:** `docs/superpowers/specs/2026-10-07-weaponshield-v2-design.md` (sections 3.4, 3.5, 4, 9)

## Global Constraints

- Branch `v2`; commits under the repo's configured identity. **No `Co-Authored-By` trailers, no "Generated with" lines.** Before every push: `git log origin/v2..HEAD --format=%B | grep -i -E "co-authored|claude|generated with"` must print nothing.
- Do not modify `web/src/core/**`, `web/src/engine/**` or `web/src/config/models.ts` except where a task says so; build on their public interfaces.
- Look: dark "security operations console" by default, light theme via toggle. Exact tokens are in Task 1; components use tokens (`bg-surface`, `text-muted`, …), never raw hex, except `OVERLAY_COLORS` on canvas.
- Signal colours carry meaning only: person green, weapon red, ARMED amber. Brand accent is cyan. Status colours are never used for chart series.
- Chart series palette (validated with the dataviz validator against our surfaces): dark `#3987e5, #d95926, #199e70, #c98500`; light `#2a78d6, #eb6834, #1baf7a, #eda100`, in that fixed order for person, pistol, rifle, knife. Light-mode contrast WARN ⇒ every chart has direct labels or a table view next to it.
- Responsive down to 360 px wide with no horizontal page scroll; side panel stacks below the stage under `lg`.
- Accessibility: keyboard reachable, visible focus rings, `prefers-reduced-motion` respected, no serious/critical axe violations.
- Alerts: banner + sound + Notification API, at most one alert per 10 s (global cooldown).
- Privacy: frames never leave the device — no network calls besides same-origin static assets.
- Fonts are self-hosted (`@fontsource-variable/*`) so COEP `require-corp` never blocks them; no third-party CDNs.
- Lighthouse ≥ 90 for performance, accessibility and best practices on `/` (checked in Task 12).

---

## File Structure

```
web/index.html                              theme bootstrap script, meta, icon
web/public/icon.svg                         app icon / favicon
web/demo.json                               demo clip manifest (download + attribution metadata)
web/scripts/prepare-assets.mjs              (modify) also downloads demo clips into public/demo/
web/vite.config.ts                          (modify) @ alias, PWA plugin, test setup
web/tsconfig.app.json                       (modify) paths + resolveJsonModule
web/src/main.tsx                            (modify) router + providers
web/src/styles/tokens.css                   design tokens + base styles (replaces index.css contents)
web/src/lib/cn.ts                           class merge helper
web/src/lib/download.ts                     downloadBlob / downloadText
web/src/lib/theme.ts                        theme read/write (localStorage-safe)
web/src/components/ui/*.tsx                 Button, Card, Badge, Progress, Segmented, Tabs, Tooltip, Kbd, Stat
web/src/components/layout/*.tsx             SiteHeader, SiteFooter, RootLayout, ThemeToggle, Logo
web/src/app/router.tsx                      route table
web/src/app/detector-context.tsx            DetectorProvider + useDetector
web/src/features/detection/rate-meter.ts    FPS EMA
web/src/features/detection/events.ts        EventRecord reducer + CSV/JSON export
web/src/features/detection/use-video-detection.ts
web/src/features/stage/Stage.tsx            video + overlay canvas + viewfinder chrome
web/src/features/stage/stage-geometry.ts    object-contain math + mirroring (pure)
web/src/features/alerts/alert-gate.ts       global cooldown (pure)
web/src/features/alerts/use-threat-alerts.ts
web/src/features/alerts/AlertBanner.tsx
web/src/features/console/*.tsx              ConsolePage, LiveMode, VideoMode, ImageMode, StatsPanel,
                                            SensitivityControl, EventTimeline, ModelGate, BackendBadge
web/src/features/console/clip-recorder.ts   MediaRecorder helper (pure mime pick + recorder)
web/src/features/demo/demo-clips.ts         typed view of demo.json
web/src/features/demo/DemoGallery.tsx
web/src/features/landing/LandingPage.tsx    + HeroDemo.tsx, HowItWorks.tsx
web/src/features/model-card/*.tsx           ModelCardPage, PrCurves, ConfusionMatrix, DatasetBars, ...
web/src/content/model-card.json             model card data (status "pending" until Plan 5)
web/src/content/model-card.ts               types + accessors
web/src/features/architecture/ArchitecturePage.tsx + PipelineDiagram.tsx
web/src/features/not-found/NotFoundPage.tsx
web/src/test/setup.ts                       jest-dom + ResizeObserver polyfill
web/e2e/*.spec.ts                           landing, console, model, a11y, responsive, isolation
```
Deleted: `web/src/lab/`, `web/src/App.tsx`, `web/src/index.css`, `web/e2e/lab.spec.ts`.

---

### Task 1: Design system, app shell and routing

**Files:**
- Modify: `web/package.json` (deps), `web/vite.config.ts`, `web/tsconfig.app.json`, `web/index.html`, `web/src/main.tsx`
- Create: `web/public/icon.svg`, `web/src/styles/tokens.css`, `web/src/lib/cn.ts`, `web/src/lib/theme.ts`, `web/src/components/ui/{button,card,badge,progress,segmented,tabs,tooltip,kbd,stat}.tsx`, `web/src/components/layout/{Logo,ThemeToggle,SiteHeader,SiteFooter,RootLayout}.tsx`, `web/src/app/router.tsx`, `web/src/features/not-found/NotFoundPage.tsx`, `web/src/test/setup.ts`
- Create (temporary route bodies, filled by later tasks): `web/src/features/landing/LandingPage.tsx`, `web/src/features/console/ConsolePage.tsx`, `web/src/features/model-card/ModelCardPage.tsx`, `web/src/features/architecture/ArchitecturePage.tsx`
- Move: `web/src/lab/LabPage.tsx` stays importable at `/lab` until Task 11 deletes it
- Delete: `web/src/App.tsx`, `web/src/index.css`
- Test: `web/src/components/layout/SiteHeader.test.tsx`, `web/src/lib/theme.test.ts`

**Interfaces:**
- Produces:
  - `cn(...inputs: ClassValue[]): string`
  - `type Theme = 'dark' | 'light'`; `readTheme(): Theme` (default `'dark'`), `writeTheme(t: Theme): void`, `applyTheme(t: Theme): void` (sets `document.documentElement.dataset.theme`)
  - UI: `Button` (`variant: 'primary'|'secondary'|'ghost'|'danger'`, `size: 'sm'|'md'|'lg'`), `buttonClasses(opts)`, `Card`, `CardHeader`, `CardTitle`, `CardBody`, `Badge` (`tone: 'neutral'|'accent'|'person'|'weapon'|'armed'|'success'`), `Progress` (`value: number` 0–1, `label: string`), `Segmented<T extends string>` (`value`, `onValueChange`, `options: {value: T; label: string}[]`, `ariaLabel`), `Tabs`, `TabsList`, `TabsTrigger`, `TabsContent`, `Tooltip` (`content`, `children`), `TooltipProvider`, `Kbd`, `Stat` (`label`, `value`, `unit?`, `hint?`, `testId?`)
  - Routes: `/` LandingPage, `/console` ConsolePage, `/model` ModelCardPage (lazy), `/architecture` ArchitecturePage, `/lab` LabPage (temporary), `*` NotFoundPage — all inside `RootLayout` (header + `<Outlet/>` + footer)
  - `@/` path alias → `web/src/`

- [ ] **Step 1: Install dependencies**

```bash
cd /d/Projects/WeaponShield-AI/web
npm install react-router@^8.4.0 @radix-ui/react-tabs@^1.1.22 @radix-ui/react-toggle-group@^1.1.20 @radix-ui/react-tooltip@^1.3.0 \
  clsx@^2.1.1 tailwind-merge@^3.7.0 lucide-react@^1.52.0 motion@^14.0.0 recharts@^3.10.1 react-is@^19.3.0 \
  @fontsource-variable/inter@^5.3.0 @fontsource-variable/jetbrains-mono@^5.3.0
npm install -D @testing-library/user-event@^14.6.7 @testing-library/jest-dom@^7.0.1 vite-plugin-pwa@^2.0.0 workbox-window@^7.4.1 @axe-core/playwright
```
Expected: exit 0. Keep `onnxruntime-web`/`onnxruntime-node` at exactly `1.30.0` and `typescript` at `~6.0.3` (check `package.json` afterwards).

- [ ] **Step 2: Path alias, JSON imports and test setup**

`web/tsconfig.app.json` — add inside `compilerOptions`:
```json
    "baseUrl": ".",
    "paths": { "@/*": ["src/*"] },
    "resolveJsonModule": true,
```

`web/vite.config.ts` — full file:
```ts
import { fileURLToPath, URL } from 'node:url';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

// Cross-origin isolation enables SharedArrayBuffer, which ONNX Runtime Web needs for WASM threads.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  server: { headers: isolation },
  preview: { headers: isolation },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  worker: { format: 'es' },
  test: {
    include: ['src/**/*.test.{ts,tsx}'],
    environment: 'node',
    setupFiles: ['src/test/setup.ts'],
    testTimeout: 30_000,
  },
});
```

`web/src/test/setup.ts`:
```ts
import '@testing-library/jest-dom/vitest';

// jsdom lacks ResizeObserver (used by Recharts' ResponsiveContainer and the Stage).
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
}
```

- [ ] **Step 3: Write the failing tests**

`web/src/lib/theme.test.ts`:
```ts
// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { applyTheme, readTheme, writeTheme } from './theme';

afterEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
});

describe('theme', () => {
  it('defaults to dark', () => {
    expect(readTheme()).toBe('dark');
  });

  it('round-trips through localStorage', () => {
    writeTheme('light');
    expect(readTheme()).toBe('light');
  });

  it('survives storage that throws', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked');
    });
    expect(() => writeTheme('light')).not.toThrow();
    expect(readTheme()).toBe('dark');
  });

  it('applies the theme to the document element', () => {
    applyTheme('light');
    expect(document.documentElement.dataset.theme).toBe('light');
  });
});
```

`web/src/components/layout/SiteHeader.test.tsx`:
```tsx
// @vitest-environment jsdom
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router';
import { describe, expect, it } from 'vitest';
import { SiteHeader } from './SiteHeader';

const renderHeader = () =>
  render(
    <MemoryRouter initialEntries={['/console']}>
      <SiteHeader />
    </MemoryRouter>,
  );

describe('SiteHeader', () => {
  it('links to every page and marks the current one', () => {
    renderHeader();
    const nav = screen.getByRole('navigation', { name: 'Main' });
    for (const name of ['Console', 'Model card', 'How it works']) {
      expect(nav).toContainElement(screen.getAllByRole('link', { name })[0]!);
    }
    expect(screen.getAllByRole('link', { name: 'Console' })[0]).toHaveAttribute('aria-current', 'page');
  });

  it('toggles the theme', async () => {
    renderHeader();
    const toggle = screen.getByRole('button', { name: /switch to light theme/i });
    await userEvent.click(toggle);
    expect(document.documentElement.dataset.theme).toBe('light');
    expect(screen.getByRole('button', { name: /switch to dark theme/i })).toBeInTheDocument();
  });
});
```

- [ ] **Step 4: Run them to verify they fail**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/lib/theme.test.ts src/components/layout/SiteHeader.test.tsx`
Expected: FAIL — cannot resolve `./theme` / `./SiteHeader`.

- [ ] **Step 5: Write the design tokens**

`web/src/styles/tokens.css`:
```css
@import "tailwindcss";
@import "@fontsource-variable/inter";
@import "@fontsource-variable/jetbrains-mono";

@custom-variant dark (&:where([data-theme="dark"], [data-theme="dark"] *));

@theme {
  --font-sans: "Inter Variable", ui-sans-serif, system-ui, sans-serif;
  --font-mono: "JetBrains Mono Variable", ui-monospace, SFMono-Regular, Menlo, monospace;

  --color-bg: var(--ws-bg);
  --color-surface: var(--ws-surface);
  --color-surface-2: var(--ws-surface-2);
  --color-border: var(--ws-border);
  --color-fg: var(--ws-fg);
  --color-muted: var(--ws-muted);
  --color-accent: var(--ws-accent);
  --color-accent-fg: var(--ws-accent-fg);
  --color-person: var(--ws-person);
  --color-weapon: var(--ws-weapon);
  --color-armed: var(--ws-armed);
  --color-success: var(--ws-success);
  --color-chart-1: var(--ws-chart-1);
  --color-chart-2: var(--ws-chart-2);
  --color-chart-3: var(--ws-chart-3);
  --color-chart-4: var(--ws-chart-4);

  --radius-card: 14px;
}

:root,
[data-theme="dark"] {
  color-scheme: dark;
  --ws-bg: #07090c;
  --ws-surface: #0d1117;
  --ws-surface-2: #131a22;
  --ws-border: #1f2a36;
  --ws-fg: #e6edf3;
  --ws-muted: #8b98a5;
  --ws-accent: #38bdf8;
  --ws-accent-fg: #04121c;
  --ws-person: #22c55e;
  --ws-weapon: #ef4444;
  --ws-armed: #f59e0b;
  --ws-success: #22c55e;
  --ws-grid: rgb(148 163 184 / 0.07);
  --ws-chart-1: #3987e5;
  --ws-chart-2: #d95926;
  --ws-chart-3: #199e70;
  --ws-chart-4: #c98500;
}

[data-theme="light"] {
  color-scheme: light;
  --ws-bg: #f5f7fa;
  --ws-surface: #ffffff;
  --ws-surface-2: #eef2f6;
  --ws-border: #d8e0e8;
  --ws-fg: #0b1220;
  --ws-muted: #556372;
  --ws-accent: #0284c7;
  --ws-accent-fg: #ffffff;
  --ws-person: #16a34a;
  --ws-weapon: #dc2626;
  --ws-armed: #d97706;
  --ws-success: #16a34a;
  --ws-grid: rgb(15 23 42 / 0.06);
  --ws-chart-1: #2a78d6;
  --ws-chart-2: #eb6834;
  --ws-chart-3: #1baf7a;
  --ws-chart-4: #eda100;
}

@layer base {
  html {
    background: var(--ws-bg);
    color: var(--ws-fg);
    font-family: var(--font-sans);
    -webkit-font-smoothing: antialiased;
  }
  body {
    min-height: 100dvh;
    background: var(--ws-bg);
  }
  :focus-visible {
    outline: 2px solid var(--ws-accent);
    outline-offset: 2px;
  }
  ::selection {
    background: color-mix(in oklab, var(--ws-accent) 35%, transparent);
  }
}

@utility bg-grid {
  background-image:
    linear-gradient(var(--ws-grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--ws-grid) 1px, transparent 1px);
  background-size: 32px 32px;
}

@utility tabular {
  font-variant-numeric: tabular-nums;
}
```

- [ ] **Step 6: Bootstrap the theme before paint and add the icon**

`web/index.html`:
```html
<!doctype html>
<html lang="en" data-theme="dark">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="description" content="Real-time weapon and person detection that runs entirely in your browser. Nothing is uploaded." />
    <meta name="theme-color" content="#07090c" />
    <link rel="icon" type="image/svg+xml" href="/icon.svg" />
    <title>WeaponShield AI — in-browser weapon detection</title>
    <script>
      try {
        var t = localStorage.getItem('ws-theme');
        if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t;
      } catch (e) {}
    </script>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`web/public/icon.svg`:
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="#07090c"/>
  <path d="M32 9 50 16v14c0 12-7.7 21.4-18 25-10.3-3.6-18-13-18-25V16z" fill="none" stroke="#38bdf8" stroke-width="4" stroke-linejoin="round"/>
  <circle cx="32" cy="31" r="7" fill="none" stroke="#38bdf8" stroke-width="4"/>
  <path d="M32 18v6M32 38v6M19 31h6M39 31h6" stroke="#38bdf8" stroke-width="4" stroke-linecap="round"/>
</svg>
```

- [ ] **Step 7: Implement helpers and UI primitives**

`web/src/lib/cn.ts`:
```ts
import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export const cn = (...inputs: ClassValue[]): string => twMerge(clsx(inputs));
```

`web/src/lib/theme.ts`:
```ts
export type Theme = 'dark' | 'light';
const KEY = 'ws-theme';

export function readTheme(): Theme {
  try {
    const v = localStorage.getItem(KEY);
    return v === 'light' ? 'light' : 'dark';
  } catch {
    return 'dark';
  }
}

export function writeTheme(t: Theme): void {
  try {
    localStorage.setItem(KEY, t);
  } catch {
    // storage unavailable (private mode / blocked): theme lasts for this page view only
  }
}

export function applyTheme(t: Theme): void {
  document.documentElement.dataset.theme = t;
}
```

`web/src/components/ui/button.tsx`:
```tsx
import type { ButtonHTMLAttributes } from 'react';
import { cn } from '@/lib/cn';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md' | 'lg';

const base =
  'inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors select-none ' +
  'disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-4 [&_svg]:shrink-0';
const variants: Record<ButtonVariant, string> = {
  primary: 'bg-accent text-accent-fg hover:bg-accent/90',
  secondary: 'border border-border bg-surface-2 text-fg hover:bg-surface-2/70',
  ghost: 'text-muted hover:bg-surface-2 hover:text-fg',
  danger: 'bg-weapon text-white hover:bg-weapon/90',
};
const sizes: Record<ButtonSize, string> = {
  sm: 'h-8 px-3 text-sm',
  md: 'h-10 px-4 text-sm',
  lg: 'h-12 px-6 text-base',
};

export function buttonClasses({ variant = 'primary', size = 'md', className }: { variant?: ButtonVariant; size?: ButtonSize; className?: string } = {}) {
  return cn(base, variants[variant], sizes[size], className);
}

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

export function Button({ variant, size, className, type = 'button', ...props }: ButtonProps) {
  return <button type={type} className={buttonClasses({ variant, size, className })} {...props} />;
}
```

`web/src/components/ui/card.tsx`:
```tsx
import type { HTMLAttributes } from 'react';
import { cn } from '@/lib/cn';

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('rounded-[var(--radius-card)] border border-border bg-surface', className)} {...props} />;
}

export function CardHeader({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('flex items-center justify-between gap-3 border-b border-border px-4 py-3', className)} {...props} />;
}

export function CardTitle({ className, ...props }: HTMLAttributes<HTMLHeadingElement>) {
  return <h2 className={cn('font-mono text-xs font-semibold uppercase tracking-[0.12em] text-muted', className)} {...props} />;
}

export function CardBody({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('p-4', className)} {...props} />;
}
```

`web/src/components/ui/badge.tsx`:
```tsx
import type { HTMLAttributes } from 'react';
import { cn } from '@/lib/cn';

export type BadgeTone = 'neutral' | 'accent' | 'person' | 'weapon' | 'armed' | 'success';

const tones: Record<BadgeTone, string> = {
  neutral: 'border-border text-muted',
  accent: 'border-accent/40 text-accent',
  person: 'border-person/40 text-person',
  weapon: 'border-weapon/50 text-weapon',
  armed: 'border-armed/50 text-armed',
  success: 'border-success/40 text-success',
};

export function Badge({ tone = 'neutral', className, ...props }: HTMLAttributes<HTMLSpanElement> & { tone?: BadgeTone }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 font-mono text-[11px] font-medium uppercase tracking-wider',
        tones[tone],
        className,
      )}
      {...props}
    />
  );
}
```

`web/src/components/ui/progress.tsx`:
```tsx
import { cn } from '@/lib/cn';

export function Progress({ value, label, className }: { value: number; label: string; className?: string }) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      className={cn('h-1.5 w-full overflow-hidden rounded-full bg-surface-2', className)}
    >
      <div className="h-full rounded-full bg-accent transition-[width] duration-300" style={{ width: `${pct}%` }} />
    </div>
  );
}
```

`web/src/components/ui/segmented.tsx`:
```tsx
import * as ToggleGroup from '@radix-ui/react-toggle-group';
import { cn } from '@/lib/cn';

export interface SegmentedProps<T extends string> {
  value: T;
  onValueChange: (value: T) => void;
  options: readonly { value: T; label: string }[];
  ariaLabel: string;
  className?: string;
}

export function Segmented<T extends string>({ value, onValueChange, options, ariaLabel, className }: SegmentedProps<T>) {
  return (
    <ToggleGroup.Root
      type="single"
      value={value}
      onValueChange={(v) => v && onValueChange(v as T)}
      aria-label={ariaLabel}
      className={cn('inline-flex w-full rounded-lg border border-border bg-surface-2 p-0.5', className)}
    >
      {options.map((o) => (
        <ToggleGroup.Item
          key={o.value}
          value={o.value}
          className="flex-1 rounded-md px-3 py-1.5 text-sm text-muted transition-colors hover:text-fg data-[state=on]:bg-surface data-[state=on]:text-fg data-[state=on]:shadow-sm"
        >
          {o.label}
        </ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  );
}
```

`web/src/components/ui/tabs.tsx`:
```tsx
import * as T from '@radix-ui/react-tabs';
import type { ComponentProps } from 'react';
import { cn } from '@/lib/cn';

export const Tabs = T.Root;

export function TabsList({ className, ...props }: ComponentProps<typeof T.List>) {
  return <T.List className={cn('inline-flex rounded-lg border border-border bg-surface p-1', className)} {...props} />;
}

export function TabsTrigger({ className, ...props }: ComponentProps<typeof T.Trigger>) {
  return (
    <T.Trigger
      className={cn(
        'inline-flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium text-muted transition-colors hover:text-fg',
        'data-[state=active]:bg-surface-2 data-[state=active]:text-fg [&_svg]:size-4',
        className,
      )}
      {...props}
    />
  );
}

export function TabsContent({ className, ...props }: ComponentProps<typeof T.Content>) {
  return <T.Content className={cn('mt-4 focus-visible:outline-none', className)} {...props} />;
}
```

`web/src/components/ui/tooltip.tsx`:
```tsx
import * as T from '@radix-ui/react-tooltip';
import type { ReactNode } from 'react';

export const TooltipProvider = T.Provider;

export function Tooltip({ content, children }: { content: ReactNode; children: ReactNode }) {
  return (
    <T.Root delayDuration={200}>
      <T.Trigger asChild>{children}</T.Trigger>
      <T.Portal>
        <T.Content sideOffset={6} className="z-50 max-w-xs rounded-md border border-border bg-surface-2 px-2.5 py-1.5 text-xs text-fg shadow-lg">
          {content}
        </T.Content>
      </T.Portal>
    </T.Root>
  );
}
```

`web/src/components/ui/kbd.tsx`:
```tsx
import type { HTMLAttributes } from 'react';
import { cn } from '@/lib/cn';

export function Kbd({ className, ...props }: HTMLAttributes<HTMLElement>) {
  return <kbd className={cn('rounded border border-border bg-surface-2 px-1.5 py-0.5 font-mono text-[11px] text-muted', className)} {...props} />;
}
```

`web/src/components/ui/stat.tsx`:
```tsx
import type { ReactNode } from 'react';
import { cn } from '@/lib/cn';

export function Stat({ label, value, unit, hint, testId, className }: { label: string; value: ReactNode; unit?: string; hint?: string; testId?: string; className?: string }) {
  return (
    <div className={cn('min-w-0', className)}>
      <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-muted">{label}</div>
      <div className="mt-1 flex items-baseline gap-1">
        <span data-testid={testId} className="tabular truncate font-mono text-2xl font-semibold text-fg">
          {value}
        </span>
        {unit && <span className="font-mono text-xs text-muted">{unit}</span>}
      </div>
      {hint && <div className="mt-0.5 text-xs text-muted">{hint}</div>}
    </div>
  );
}
```

- [ ] **Step 8: Implement the layout and routing**

`web/src/components/layout/Logo.tsx`:
```tsx
export function Logo({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 64" aria-hidden="true" className={className}>
      <path d="M32 9 50 16v14c0 12-7.7 21.4-18 25-10.3-3.6-18-13-18-25V16z" fill="none" stroke="currentColor" strokeWidth="4" strokeLinejoin="round" />
      <circle cx="32" cy="31" r="7" fill="none" stroke="currentColor" strokeWidth="4" />
      <path d="M32 18v6M32 38v6M19 31h6M39 31h6" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
    </svg>
  );
}
```

`web/src/components/layout/ThemeToggle.tsx`:
```tsx
import { Moon, Sun } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { applyTheme, readTheme, type Theme, writeTheme } from '@/lib/theme';

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => readTheme());
  const next: Theme = theme === 'dark' ? 'light' : 'dark';
  return (
    <Button
      variant="ghost"
      size="sm"
      aria-label={`Switch to ${next} theme`}
      onClick={() => {
        applyTheme(next);
        writeTheme(next);
        setTheme(next);
      }}
    >
      {theme === 'dark' ? <Sun /> : <Moon />}
    </Button>
  );
}
```

`web/src/components/layout/SiteHeader.tsx`:
```tsx
import { Github, Menu, X } from 'lucide-react';
import { useState } from 'react';
import { Link, NavLink } from 'react-router';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/cn';
import { Logo } from './Logo';
import { ThemeToggle } from './ThemeToggle';

const LINKS = [
  { to: '/console', label: 'Console' },
  { to: '/model', label: 'Model card' },
  { to: '/architecture', label: 'How it works' },
] as const;

export const REPO_URL = 'https://github.com/Halok600/WeaponShield-AI';

const linkClass = ({ isActive }: { isActive: boolean }) =>
  cn('rounded-md px-3 py-1.5 text-sm transition-colors', isActive ? 'bg-surface-2 text-fg' : 'text-muted hover:text-fg');

export function SiteHeader() {
  const [open, setOpen] = useState(false);
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-bg/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-3 px-4">
        <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight" onClick={() => setOpen(false)}>
          <Logo className="size-6 text-accent" />
          <span>WeaponShield</span>
          <span className="hidden font-mono text-[11px] text-muted sm:inline">v2</span>
        </Link>
        <nav aria-label="Main" className="ml-4 hidden items-center gap-1 md:flex">
          {LINKS.map((l) => (
            <NavLink key={l.to} to={l.to} className={linkClass}>
              {l.label}
            </NavLink>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-1">
          <a href={REPO_URL} target="_blank" rel="noreferrer" aria-label="Source code on GitHub" className="rounded-md p-2 text-muted hover:text-fg">
            <Github className="size-4" />
          </a>
          <ThemeToggle />
          <Button variant="ghost" size="sm" className="md:hidden" aria-label={open ? 'Close menu' : 'Open menu'} aria-expanded={open} onClick={() => setOpen((v) => !v)}>
            {open ? <X /> : <Menu />}
          </Button>
        </div>
      </div>
      {open && (
        <nav aria-label="Mobile" className="border-t border-border px-4 py-2 md:hidden">
          {LINKS.map((l) => (
            <NavLink key={l.to} to={l.to} className={(s) => cn(linkClass(s), 'block')} onClick={() => setOpen(false)}>
              {l.label}
            </NavLink>
          ))}
        </nav>
      )}
    </header>
  );
}
```

`web/src/components/layout/SiteFooter.tsx`:
```tsx
import { Link } from 'react-router';
import { REPO_URL } from './SiteHeader';

export function SiteFooter() {
  return (
    <footer className="mt-16 border-t border-border">
      <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-6 text-sm text-muted sm:flex-row sm:items-center sm:justify-between">
        <p>
          WeaponShield AI · AGPL-3.0 · frames are processed on your device and never uploaded.
        </p>
        <div className="flex gap-4">
          <Link to="/model" className="hover:text-fg">Model card</Link>
          <Link to="/architecture" className="hover:text-fg">How it works</Link>
          <a href={REPO_URL} target="_blank" rel="noreferrer" className="hover:text-fg">GitHub</a>
        </div>
      </div>
    </footer>
  );
}
```

`web/src/components/layout/RootLayout.tsx`:
```tsx
import { Outlet, ScrollRestoration } from 'react-router';
import { TooltipProvider } from '@/components/ui/tooltip';
import { SiteFooter } from './SiteFooter';
import { SiteHeader } from './SiteHeader';

export function RootLayout() {
  return (
    <TooltipProvider>
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-3 focus:z-50 focus:rounded-md focus:bg-surface focus:px-3 focus:py-2">
        Skip to content
      </a>
      <SiteHeader />
      <main id="main" className="mx-auto w-full max-w-7xl px-4">
        <Outlet />
      </main>
      <SiteFooter />
      <ScrollRestoration />
    </TooltipProvider>
  );
}
```

`web/src/features/not-found/NotFoundPage.tsx`:
```tsx
import { Link } from 'react-router';
import { buttonClasses } from '@/components/ui/button';

export function NotFoundPage() {
  return (
    <section className="py-24 text-center">
      <p className="font-mono text-sm text-muted">404 · NO SIGNAL</p>
      <h1 className="mt-3 text-3xl font-semibold tracking-tight">This page doesn’t exist</h1>
      <Link to="/" className={buttonClasses({ className: 'mt-8' })}>Back to home</Link>
    </section>
  );
}
```

Temporary page bodies (each later task replaces its file entirely):

`web/src/features/landing/LandingPage.tsx`:
```tsx
export function LandingPage() {
  return <h1 className="py-16 text-3xl font-semibold">WeaponShield</h1>;
}
```
`web/src/features/console/ConsolePage.tsx`:
```tsx
export function ConsolePage() {
  return <h1 className="py-16 text-3xl font-semibold">Detection console</h1>;
}
```
`web/src/features/model-card/ModelCardPage.tsx`:
```tsx
export function ModelCardPage() {
  return <h1 className="py-16 text-3xl font-semibold">Model card</h1>;
}
```
`web/src/features/architecture/ArchitecturePage.tsx`:
```tsx
export function ArchitecturePage() {
  return <h1 className="py-16 text-3xl font-semibold">How it works</h1>;
}
```

`web/src/app/router.tsx`:
```tsx
import { createBrowserRouter } from 'react-router';
import { RootLayout } from '@/components/layout/RootLayout';
import { ArchitecturePage } from '@/features/architecture/ArchitecturePage';
import { ConsolePage } from '@/features/console/ConsolePage';
import { LandingPage } from '@/features/landing/LandingPage';
import { NotFoundPage } from '@/features/not-found/NotFoundPage';
import { LabPage } from '@/lab/LabPage';

export const router = createBrowserRouter([
  {
    path: '/',
    Component: RootLayout,
    children: [
      { index: true, Component: LandingPage },
      { path: 'console', Component: ConsolePage },
      {
        path: 'model',
        lazy: async () => ({ Component: (await import('@/features/model-card/ModelCardPage')).ModelCardPage }),
      },
      { path: 'architecture', Component: ArchitecturePage },
      { path: 'lab', Component: LabPage },
      { path: '*', Component: NotFoundPage },
    ],
  },
]);
```

`web/src/main.tsx`:
```tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { RouterProvider } from 'react-router/dom';
import { router } from '@/app/router';
import '@/styles/tokens.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
```

Delete `web/src/App.tsx` and `web/src/index.css`. In `web/e2e/lab.spec.ts` change `page.goto('/')` to `page.goto('/lab')` so the existing e2e keeps passing until Task 11.

- [ ] **Step 9: Run tests, typecheck, lint, build and e2e**

```bash
cd /d/Projects/WeaponShield-AI/web
npx vitest run src/lib/theme.test.ts src/components/layout/SiteHeader.test.tsx
npm test && npm run typecheck && npm run lint && npm run build && npm run e2e
```
Expected: all pass (lab e2e now under `/lab`). Then `npm run preview`, open `/`, `/console`, `/model`, `/architecture`, `/nope` and the theme toggle with the Playwright MCP or a browser; confirm the header/footer render in both themes at 1280 px and 360 px wide.

- [ ] **Step 10: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web
git commit -m "feat(web): design tokens, UI primitives, app shell and routes"
```

---

### Task 2: Detector provider

**Files:**
- Create: `web/src/app/detector-context.tsx`
- Modify: `web/src/components/layout/RootLayout.tsx` (wrap with `DetectorProvider`)
- Test: `web/src/app/detector-context.test.tsx`

**Interfaces:**
- Consumes: `DetectorClient`, `createDetectorWorker`, `ReadyInfo`, `Backend` (Plan 1); `ACTIVE_MODEL`, `ModelConfig`
- Produces:
  - `type DetectorStatus = 'idle' | 'loading' | 'ready' | 'error'`
  - `interface DetectorContextValue { status: DetectorStatus; backend: Backend | null; progress: number; error: string | null; loadMs: number | null; client: DetectorClient | null; ensureReady(): void; retry(): void }`
  - `DetectorProvider({ children, createClient?, model? })` — creates the client lazily on the first `ensureReady()`; `retry()` disposes and starts over; disposes on unmount
  - `useDetector(): DetectorContextValue` (throws if used outside the provider)

- [ ] **Step 1: Write the failing test**

`web/src/app/detector-context.test.tsx`:
```tsx
// @vitest-environment jsdom
import { act, render, screen } from '@testing-library/react';
import { useEffect } from 'react';
import { describe, expect, it, vi } from 'vitest';
import type { DetectorClient } from '@/engine/detector-client';
import { DetectorProvider, useDetector } from './detector-context';

function fakeClient(result: () => Promise<{ backend: 'wasm' | 'webgpu'; loadMs: number }>) {
  const c = { onProgress: null as null | ((l: number, t: number) => void), init: vi.fn(result), dispose: vi.fn() };
  return c as unknown as DetectorClient & { onProgress: (l: number, t: number) => void; dispose: ReturnType<typeof vi.fn> };
}

function Probe({ start = true }: { start?: boolean }) {
  const d = useDetector();
  useEffect(() => {
    if (start) d.ensureReady();
  }, [d, start]);
  return (
    <div>
      <span data-testid="status">{d.status}</span>
      <span data-testid="backend">{d.backend ?? '-'}</span>
      <span data-testid="progress">{d.progress.toFixed(2)}</span>
      <span data-testid="error">{d.error ?? '-'}</span>
      <button onClick={d.retry}>retry</button>
    </div>
  );
}

describe('DetectorProvider', () => {
  it('stays idle until ensureReady is called', () => {
    const create = vi.fn();
    render(
      <DetectorProvider createClient={create}>
        <Probe start={false} />
      </DetectorProvider>,
    );
    expect(screen.getByTestId('status')).toHaveTextContent('idle');
    expect(create).not.toHaveBeenCalled();
  });

  it('loads once, reports progress and the backend', async () => {
    let resolve!: (v: { backend: 'wasm'; loadMs: number }) => void;
    const pending = new Promise<{ backend: 'wasm'; loadMs: number }>((r) => (resolve = r));
    const client = fakeClient(() => pending);
    const create = vi.fn(() => client);
    render(
      <DetectorProvider createClient={create}>
        <Probe />
        <Probe />
      </DetectorProvider>,
    );
    expect(screen.getAllByTestId('status')[0]).toHaveTextContent('loading');
    act(() => client.onProgress(5, 10));
    expect(screen.getAllByTestId('progress')[0]).toHaveTextContent('0.50');
    await act(async () => resolve({ backend: 'wasm', loadMs: 12 }));
    expect(screen.getAllByTestId('status')[0]).toHaveTextContent('ready');
    expect(screen.getAllByTestId('backend')[0]).toHaveTextContent('wasm');
    expect(create).toHaveBeenCalledTimes(1);
  });

  it('surfaces errors and retries with a fresh client', async () => {
    const failing = fakeClient(() => Promise.reject(new Error('HTTP 404')));
    const working = fakeClient(() => Promise.resolve({ backend: 'webgpu', loadMs: 3 }));
    const create = vi.fn().mockReturnValueOnce(failing).mockReturnValueOnce(working);
    render(
      <DetectorProvider createClient={create}>
        <Probe />
      </DetectorProvider>,
    );
    await act(async () => {});
    expect(screen.getByTestId('status')).toHaveTextContent('error');
    expect(screen.getByTestId('error')).toHaveTextContent('HTTP 404');
    await act(async () => screen.getByText('retry').click());
    expect(failing.dispose).toHaveBeenCalled();
    expect(screen.getByTestId('status')).toHaveTextContent('ready');
    expect(screen.getByTestId('backend')).toHaveTextContent('webgpu');
  });

  it('disposes the client on unmount', async () => {
    const client = fakeClient(() => Promise.resolve({ backend: 'wasm', loadMs: 1 }));
    const { unmount } = render(
      <DetectorProvider createClient={() => client}>
        <Probe />
      </DetectorProvider>,
    );
    await act(async () => {});
    unmount();
    expect(client.dispose).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/app/detector-context.test.tsx`
Expected: FAIL — cannot resolve `./detector-context`.

- [ ] **Step 3: Implement `web/src/app/detector-context.tsx`**

```tsx
import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { ACTIVE_MODEL, type ModelConfig } from '@/config/models';
import { createDetectorWorker, DetectorClient } from '@/engine/detector-client';
import type { Backend } from '@/engine/protocol';

export type DetectorStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface DetectorContextValue {
  status: DetectorStatus;
  backend: Backend | null;
  progress: number;
  error: string | null;
  loadMs: number | null;
  client: DetectorClient | null;
  ensureReady(): void;
  retry(): void;
}

interface State {
  status: DetectorStatus;
  backend: Backend | null;
  progress: number;
  error: string | null;
  loadMs: number | null;
  client: DetectorClient | null;
}

const INITIAL: State = { status: 'idle', backend: null, progress: 0, error: null, loadMs: null, client: null };
const DetectorContext = createContext<DetectorContextValue | null>(null);

const defaultCreate = () => new DetectorClient(createDetectorWorker());

export function DetectorProvider({
  children,
  createClient = defaultCreate,
  model = ACTIVE_MODEL,
}: {
  children: ReactNode;
  createClient?: () => DetectorClient;
  model?: ModelConfig;
}) {
  const [state, setState] = useState<State>(INITIAL);
  const clientRef = useRef<DetectorClient | null>(null);
  const generation = useRef(0);

  const start = useCallback(() => {
    const gen = ++generation.current;
    const client = createClient();
    clientRef.current = client;
    client.onProgress = (loaded, total) => {
      if (gen === generation.current) setState((s) => ({ ...s, progress: total > 0 ? loaded / total : s.progress }));
    };
    setState({ ...INITIAL, status: 'loading' });
    client.init(model).then(
      (info) => {
        if (gen === generation.current) setState((s) => ({ ...s, status: 'ready', backend: info.backend, loadMs: info.loadMs, progress: 1, client }));
      },
      (e: unknown) => {
        if (gen === generation.current) setState((s) => ({ ...s, status: 'error', error: e instanceof Error ? e.message : String(e) }));
      },
    );
  }, [createClient, model]);

  const ensureReady = useCallback(() => {
    if (clientRef.current === null) start();
  }, [start]);

  const retry = useCallback(() => {
    clientRef.current?.dispose();
    clientRef.current = null;
    start();
  }, [start]);

  useEffect(
    () => () => {
      generation.current++;
      clientRef.current?.dispose();
      clientRef.current = null;
    },
    [],
  );

  const value = useMemo<DetectorContextValue>(() => ({ ...state, ensureReady, retry }), [state, ensureReady, retry]);
  return <DetectorContext.Provider value={value}>{children}</DetectorContext.Provider>;
}

export function useDetector(): DetectorContextValue {
  const v = useContext(DetectorContext);
  if (!v) throw new Error('useDetector must be used inside <DetectorProvider>');
  return v;
}
```

Note on React StrictMode: the unmount cleanup above runs once in StrictMode's simulated remount, disposing nothing because no client exists before `ensureReady()` is first called from an effect; the subsequent effect re-run calls `ensureReady()` again and creates the client. The `generation` counter discards results from any disposed client.

- [ ] **Step 4: Wrap the app**

In `web/src/components/layout/RootLayout.tsx`, import `DetectorProvider` from `@/app/detector-context` and wrap the whole returned tree (`<DetectorProvider><TooltipProvider>…</TooltipProvider></DetectorProvider>`).

- [ ] **Step 5: Run tests and checks**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/app/detector-context.test.tsx && npm test && npm run typecheck && npm run lint`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/src/app web/src/components/layout/RootLayout.tsx
git commit -m "feat(web): app-wide DetectorProvider with lazy model load, progress and retry"
```

---

### Task 3: Stage, geometry, event log and the video detection hook

**Files:**
- Create: `web/src/features/stage/stage-geometry.ts`, `web/src/features/stage/Stage.tsx`, `web/src/features/detection/rate-meter.ts`, `web/src/features/detection/events.ts`, `web/src/features/detection/use-video-detection.ts`
- Test: `web/src/features/stage/stage-geometry.test.ts`, `web/src/features/detection/rate-meter.test.ts`, `web/src/features/detection/events.test.ts`

**Interfaces:**
- Consumes: `Pipeline`, `PipelineFrame`, `Preset` (Plan 1); `ThreatEvent`, `ThreatFrame`; `Detection`; `startLiveLoop`, `DetectResult`; `buildTrackPrimitives`, `buildDetectionPrimitives`, `paint`, `Primitive`; `useDetector`
- Produces:
  - `interface ContentRect { x: number; y: number; width: number; height: number; scale: number }`
  - `containRect(boxW: number, boxH: number, srcW: number, srcH: number): ContentRect`
  - `mirrorPrimitives(prims: readonly Primitive[], srcW: number): Primitive[]`
  - `class RateMeter { constructor(alpha?: number); tick(nowMs: number): number; reset(): void }`
  - `interface EventRecord extends ThreatEvent { snapshot: string | null }`
  - `applyThreatFrame(records: readonly EventRecord[], threat: ThreatFrame, snapshot: (e: ThreatEvent) => string | null, max?: number): EventRecord[]` (returns the same array instance when nothing changed)
  - `eventsToCsv(records: readonly EventRecord[]): string`, `eventsToJson(records: readonly EventRecord[]): string`
  - `interface StageHandle { drawFrame(frame: PipelineFrame | null): void; drawDetections(dets: readonly Detection[]): void; snapshot(width?: number): string | null; composite(target: HTMLCanvasElement): void }`
  - `<Stage ref video={…} mirrored label live timestamp?>` — renders a `<video>` slot and the overlay canvas with viewfinder chrome; `ref` is `Ref<StageHandle>`
  - `interface LiveStats { fps: number; inferMs: number; persons: number; weapons: number; armed: number; frames: number }`
  - `useVideoDetection({ video, active, preset, clock, stage }): { stats: LiveStats; events: EventRecord[]; activeThreats: ThreatEvent[]; error: string | null; reset(): void }` — `clock: 'wall' | 'media'`; `stage: RefObject<StageHandle | null>`; stats and events update at most every 250 ms; overlay is painted every frame

- [ ] **Step 1: Write the failing tests**

`web/src/features/stage/stage-geometry.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import type { Primitive } from '@/render/overlay';
import { containRect, mirrorPrimitives } from './stage-geometry';

describe('containRect', () => {
  it('letterboxes a wide source into a tall box', () => {
    expect(containRect(400, 400, 1280, 720)).toEqual({ x: 0, y: 87.5, width: 400, height: 225, scale: 400 / 1280 });
  });
  it('pillarboxes a tall source into a wide box', () => {
    const r = containRect(1000, 500, 720, 1280);
    expect(r.height).toBe(500);
    expect(r.x).toBeCloseTo((1000 - 281.25) / 2);
  });
  it('handles a zero-size source', () => {
    expect(containRect(100, 100, 0, 0)).toEqual({ x: 0, y: 0, width: 0, height: 0, scale: 0 });
  });
});

describe('mirrorPrimitives', () => {
  it('flips boxes and trails horizontally around the source width', () => {
    const prims: Primitive[] = [
      { kind: 'box', box: { x1: 10, y1: 5, x2: 30, y2: 25 }, color: 'c', label: 'l', solid: true },
      { kind: 'trail', points: [{ x: 10, y: 1 }, { x: 20, y: 2 }], color: 'c' },
    ];
    expect(mirrorPrimitives(prims, 100)).toEqual([
      { kind: 'box', box: { x1: 70, y1: 5, x2: 90, y2: 25 }, color: 'c', label: 'l', solid: true },
      { kind: 'trail', points: [{ x: 90, y: 1 }, { x: 80, y: 2 }], color: 'c' },
    ]);
  });
});
```

`web/src/features/detection/rate-meter.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { RateMeter } from './rate-meter';

describe('RateMeter', () => {
  it('reports 0 until two ticks, then the smoothed rate', () => {
    const m = new RateMeter(0.5);
    expect(m.tick(0)).toBe(0);
    expect(m.tick(100)).toBeCloseTo(10);
    expect(m.tick(150)).toBeCloseTo(15); // 10 + 0.5 * (20 - 10)
  });
  it('ignores non-increasing timestamps and resets', () => {
    const m = new RateMeter();
    m.tick(0);
    m.tick(50);
    expect(m.tick(50)).toBeCloseTo(20);
    m.reset();
    expect(m.tick(1000)).toBe(0);
  });
});
```

`web/src/features/detection/events.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import type { ThreatEvent, ThreatFrame } from '@/core/threat';
import { applyThreatFrame, type EventRecord, eventsToCsv, eventsToJson } from './events';

const ev = (id: number, over: Partial<ThreatEvent> = {}): ThreatEvent => ({
  id,
  weaponTrackId: 10 + id,
  cls: 'pistol',
  personTrackId: 1,
  startMs: 1000,
  endMs: 1000,
  maxConf: 0.8,
  ended: false,
  ...over,
});
const frame = (over: Partial<ThreatFrame>): ThreatFrame => ({
  confirmedWeaponIds: [],
  armedPersonIds: [],
  weaponOwner: new Map(),
  active: [],
  started: [],
  ended: [],
  ...over,
});

describe('applyThreatFrame', () => {
  it('prepends started events with their snapshot', () => {
    const out = applyThreatFrame([], frame({ started: [ev(1)], active: [ev(1)] }), (e) => `snap-${e.id}`);
    expect(out).toEqual([{ ...ev(1), snapshot: 'snap-1' }]);
  });

  it('updates active and ended events and keeps unrelated records', () => {
    const start: EventRecord[] = [{ ...ev(2), snapshot: null }, { ...ev(1), snapshot: 's1' }];
    const out = applyThreatFrame(start, frame({ active: [ev(2, { endMs: 3000, maxConf: 0.95 })], ended: [ev(1, { ended: true, endMs: 1500 })] }), () => null);
    expect(out[0]).toMatchObject({ id: 2, endMs: 3000, maxConf: 0.95, snapshot: null });
    expect(out[1]).toMatchObject({ id: 1, ended: true, endMs: 1500, snapshot: 's1' });
  });

  it('returns the same array when nothing changed', () => {
    const start: EventRecord[] = [{ ...ev(1), snapshot: null }];
    expect(applyThreatFrame(start, frame({ active: [ev(1)] }), () => null)).toBe(start);
  });

  it('caps the log length', () => {
    const many = Array.from({ length: 5 }, (_, i) => ev(i + 1));
    const out = applyThreatFrame([], frame({ started: many }), () => null, 3);
    expect(out.map((r) => r.id)).toEqual([5, 4, 3]);
  });
});

describe('exports', () => {
  const records: EventRecord[] = [{ ...ev(1, { endMs: 3500, ended: true, maxConf: 0.912 }), snapshot: 'data:x' }];

  it('writes CSV with seconds and no snapshots', () => {
    expect(eventsToCsv(records)).toBe(
      'event_id,class,person_track_id,start_s,end_s,duration_s,max_confidence,ended\n1,pistol,1,1.00,3.50,2.50,0.912,true\n',
    );
  });

  it('writes JSON without snapshots', () => {
    const parsed = JSON.parse(eventsToJson(records));
    expect(parsed).toEqual([{ id: 1, class: 'pistol', personTrackId: 1, startS: 1, endS: 3.5, maxConfidence: 0.912, ended: true }]);
  });
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/features/stage src/features/detection`
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement the pure modules**

`web/src/features/stage/stage-geometry.ts`:
```ts
import type { Primitive } from '@/render/overlay';

export interface ContentRect {
  x: number;
  y: number;
  width: number;
  height: number;
  /** Displayed pixels per source pixel. */
  scale: number;
}

/** Where a source of srcW x srcH lands inside a boxW x boxH element with object-fit: contain. */
export function containRect(boxW: number, boxH: number, srcW: number, srcH: number): ContentRect {
  if (srcW <= 0 || srcH <= 0) return { x: 0, y: 0, width: 0, height: 0, scale: 0 };
  const scale = Math.min(boxW / srcW, boxH / srcH);
  const width = srcW * scale;
  const height = srcH * scale;
  return { x: (boxW - width) / 2, y: (boxH - height) / 2, width, height, scale };
}

/** Mirror overlay geometry so it matches a CSS-mirrored (selfie) video while text stays readable. */
export function mirrorPrimitives(prims: readonly Primitive[], srcW: number): Primitive[] {
  return prims.map((p) =>
    p.kind === 'box'
      ? { ...p, box: { x1: srcW - p.box.x2, y1: p.box.y1, x2: srcW - p.box.x1, y2: p.box.y2 } }
      : { ...p, points: p.points.map((pt) => ({ x: srcW - pt.x, y: pt.y })) },
  );
}
```

`web/src/features/detection/rate-meter.ts`:
```ts
/** Exponentially smoothed events-per-second from timestamps in milliseconds. */
export class RateMeter {
  private last: number | null = null;
  private ema = 0;

  constructor(private readonly alpha = 0.2) {}

  tick(nowMs: number): number {
    if (this.last !== null && nowMs > this.last) {
      const inst = 1000 / (nowMs - this.last);
      this.ema = this.ema === 0 ? inst : this.ema + this.alpha * (inst - this.ema);
    }
    if (this.last === null || nowMs > this.last) this.last = nowMs;
    return this.ema;
  }

  reset(): void {
    this.last = null;
    this.ema = 0;
  }
}
```

`web/src/features/detection/events.ts`:
```ts
import type { ThreatEvent, ThreatFrame } from '@/core/threat';

export interface EventRecord extends ThreatEvent {
  /** JPEG data URL captured when the event started, or null if capture failed. */
  snapshot: string | null;
}

export function applyThreatFrame(
  records: readonly EventRecord[],
  threat: ThreatFrame,
  snapshot: (e: ThreatEvent) => string | null,
  max = 100,
): EventRecord[] {
  let changed = false;
  let next = records as EventRecord[];
  if (threat.started.length > 0) {
    const fresh = [...threat.started].reverse().map((e) => ({ ...e, snapshot: snapshot(e) }));
    next = [...fresh, ...next];
    changed = true;
  }
  const updates = new Map<number, ThreatEvent>();
  for (const e of threat.active) updates.set(e.id, e);
  for (const e of threat.ended) updates.set(e.id, e);
  if (updates.size > 0) {
    next = next.map((r) => {
      const u = updates.get(r.id);
      if (!u || (u.endMs === r.endMs && u.maxConf === r.maxConf && u.ended === r.ended && u.cls === r.cls && u.personTrackId === r.personTrackId)) return r;
      changed = true;
      return { ...r, ...u, snapshot: r.snapshot };
    });
  }
  if (!changed) return records as EventRecord[];
  return next.length > max ? next.slice(0, max) : next;
}

const s = (ms: number) => (ms / 1000).toFixed(2);

export function eventsToCsv(records: readonly EventRecord[]): string {
  const header = 'event_id,class,person_track_id,start_s,end_s,duration_s,max_confidence,ended\n';
  return (
    header +
    records
      .map((r) => [r.id, r.cls, r.personTrackId ?? '', s(r.startMs), s(r.endMs), s(r.endMs - r.startMs), r.maxConf.toFixed(3), r.ended].join(',') + '\n')
      .join('')
  );
}

export function eventsToJson(records: readonly EventRecord[]): string {
  return JSON.stringify(
    records.map((r) => ({
      id: r.id,
      class: r.cls,
      personTrackId: r.personTrackId,
      startS: r.startMs / 1000,
      endS: r.endMs / 1000,
      maxConfidence: Number(r.maxConf.toFixed(3)),
      ended: r.ended,
    })),
    null,
    2,
  );
}
```

- [ ] **Step 4: Run the pure tests**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/features/stage src/features/detection`
Expected: PASS (9 tests).

- [ ] **Step 5: Implement `web/src/features/stage/Stage.tsx`**

```tsx
import { forwardRef, type ReactNode, useEffect, useImperativeHandle, useRef } from 'react';
import type { PipelineFrame } from '@/core/pipeline';
import type { Detection } from '@/core/types';
import { cn } from '@/lib/cn';
import { buildDetectionPrimitives, buildTrackPrimitives, paint, type Primitive } from '@/render/overlay';
import { containRect, mirrorPrimitives } from './stage-geometry';

export interface StageHandle {
  drawFrame(frame: PipelineFrame | null): void;
  drawDetections(dets: readonly Detection[]): void;
  /** JPEG data URL of the current video frame with overlay, `width` px wide. */
  snapshot(width?: number): string | null;
  /** Draw video + overlay at native resolution into `target` (used for recording/export). */
  composite(target: HTMLCanvasElement): void;
}

interface StageProps {
  video: ReactNode;
  videoEl: () => HTMLVideoElement | null;
  label: string;
  live?: boolean;
  mirrored?: boolean;
  children?: ReactNode;
  className?: string;
}

const Corner = ({ className }: { className: string }) => (
  <span aria-hidden="true" className={cn('pointer-events-none absolute size-5 border-accent/70', className)} />
);

export const Stage = forwardRef<StageHandle, StageProps>(function Stage({ video, videoEl, label, live, mirrored, children, className }, ref) {
  const boxRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastPrims = useRef<Primitive[]>([]);

  const redraw = () => {
    const canvas = canvasRef.current;
    const box = boxRef.current;
    const v = videoEl();
    if (!canvas || !box || !v) return;
    const dpr = window.devicePixelRatio || 1;
    const w = box.clientWidth;
    const h = box.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const r = containRect(w, h, v.videoWidth, v.videoHeight);
    if (r.scale === 0) return;
    ctx.setTransform(dpr, 0, 0, dpr, r.x * dpr, r.y * dpr);
    const prims = mirrored ? mirrorPrimitives(lastPrims.current, v.videoWidth) : lastPrims.current;
    ctx.save();
    paint(ctx, prims, r.scale, r.scale);
    ctx.restore();
  };

  useEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    const ro = new ResizeObserver(() => redraw());
    ro.observe(box);
    return () => ro.disconnect();
  });

  useImperativeHandle(ref, () => ({
    drawFrame(frame) {
      lastPrims.current = frame ? buildTrackPrimitives(frame) : [];
      redraw();
    },
    drawDetections(dets) {
      lastPrims.current = buildDetectionPrimitives(dets);
      redraw();
    },
    snapshot(width = 320) {
      const v = videoEl();
      if (!v || v.videoWidth === 0) return null;
      const scale = width / v.videoWidth;
      const c = document.createElement('canvas');
      c.width = width;
      c.height = Math.round(v.videoHeight * scale);
      const ctx = c.getContext('2d');
      if (!ctx) return null;
      ctx.drawImage(v, 0, 0, c.width, c.height);
      ctx.save();
      paint(ctx, lastPrims.current, scale, scale);
      ctx.restore();
      try {
        return c.toDataURL('image/jpeg', 0.75);
      } catch {
        return null;
      }
    },
    composite(target) {
      const v = videoEl();
      if (!v || v.videoWidth === 0) return;
      if (target.width !== v.videoWidth) target.width = v.videoWidth;
      if (target.height !== v.videoHeight) target.height = v.videoHeight;
      const ctx = target.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(v, 0, 0);
      ctx.save();
      paint(ctx, lastPrims.current, 1, 1);
      ctx.restore();
    },
  }));

  return (
    <div className={cn('relative overflow-hidden rounded-[var(--radius-card)] border border-border bg-black', className)}>
      <div ref={boxRef} className="relative aspect-video w-full [&_video]:absolute [&_video]:inset-0 [&_video]:size-full [&_video]:object-contain">
        <div className={cn('absolute inset-0', mirrored && '[&_video]:-scale-x-100')}>{video}</div>
        <canvas ref={canvasRef} aria-hidden="true" className="pointer-events-none absolute inset-0 size-full" />
        <Corner className="left-3 top-3 border-l-2 border-t-2" />
        <Corner className="right-3 top-3 border-r-2 border-t-2" />
        <Corner className="bottom-3 left-3 border-b-2 border-l-2" />
        <Corner className="bottom-3 right-3 border-b-2 border-r-2" />
        <div className="pointer-events-none absolute left-5 top-4 flex items-center gap-2 rounded bg-black/60 px-2 py-1 font-mono text-[11px] uppercase tracking-wider text-white">
          {live && <span className="size-1.5 animate-pulse rounded-full bg-weapon motion-reduce:animate-none" />}
          {label}
        </div>
        {children && <div className="absolute inset-0 flex items-center justify-center">{children}</div>}
      </div>
    </div>
  );
});
```

- [ ] **Step 6: Implement `web/src/features/detection/use-video-detection.ts`**

```ts
import { type RefObject, useCallback, useEffect, useRef, useState } from 'react';
import { useDetector } from '@/app/detector-context';
import { Pipeline } from '@/core/pipeline';
import type { Preset } from '@/core/presets';
import type { ThreatEvent } from '@/core/threat';
import { isWeapon } from '@/core/types';
import { startLiveLoop } from '@/engine/live-loop';
import type { StageHandle } from '@/features/stage/Stage';
import { applyThreatFrame, type EventRecord } from './events';
import { RateMeter } from './rate-meter';

export interface LiveStats {
  fps: number;
  inferMs: number;
  persons: number;
  weapons: number;
  armed: number;
  frames: number;
}

const EMPTY: LiveStats = { fps: 0, inferMs: 0, persons: 0, weapons: 0, armed: 0, frames: 0 };
const PUBLISH_MS = 250;

export interface UseVideoDetection {
  video: HTMLVideoElement | null;
  active: boolean;
  preset: Preset;
  clock: 'wall' | 'media';
  stage: RefObject<StageHandle | null>;
}

export function useVideoDetection({ video, active, preset, clock, stage }: UseVideoDetection) {
  const { client, status } = useDetector();
  const [stats, setStats] = useState<LiveStats>(EMPTY);
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [activeThreats, setActiveThreats] = useState<ThreatEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const pipeline = useRef(new Pipeline(preset));
  const presetRef = useRef(preset);

  useEffect(() => {
    presetRef.current = preset;
    pipeline.current.setPreset(preset);
  }, [preset]);

  const reset = useCallback(() => {
    pipeline.current.reset();
    setEvents([]);
    setActiveThreats([]);
    setStats(EMPTY);
    stage.current?.drawFrame(null);
  }, [stage]);

  useEffect(() => {
    if (!active || !video || !client || status !== 'ready') return;
    const meter = new RateMeter();
    let frames = 0;
    let lastPublish = 0;
    setError(null);

    const stop = startLiveLoop({
      video,
      client,
      lowConf: () => presetRef.current.lowConf,
      clock: clock === 'media' ? () => video.currentTime * 1000 : () => performance.now(),
      onResult: (r, t) => {
        const frame = pipeline.current.process(r.detections, t);
        stage.current?.drawFrame(frame);
        frames++;
        const fps = meter.tick(performance.now());
        const now = performance.now();
        const publish = now - lastPublish >= PUBLISH_MS || frame.threat.started.length > 0 || frame.threat.ended.length > 0;
        if (frame.threat.started.length || frame.threat.ended.length || (publish && frame.threat.active.length)) {
          // Capture thumbnails now, while the video still shows the frame that started the event.
          const snaps = new Map(frame.threat.started.map((e) => [e.id, stage.current?.snapshot() ?? null] as const));
          const threat = publish ? frame.threat : { ...frame.threat, active: [] };
          setEvents((prev) => applyThreatFrame(prev, threat, (e) => snaps.get(e.id) ?? null));
        }
        if (publish) {
          lastPublish = now;
          const confirmed = new Set(frame.threat.confirmedWeaponIds);
          setStats({
            fps,
            inferMs: r.inferMs,
            persons: frame.tracks.filter((tr) => tr.cls === 'person').length,
            weapons: frame.tracks.filter((tr) => isWeapon(tr.cls) && confirmed.has(tr.id)).length,
            armed: frame.threat.armedPersonIds.length,
            frames,
          });
          setActiveThreats(frame.threat.active);
        }
      },
      onError: (e) => setError(e instanceof Error ? e.message : String(e)),
    });
    return () => stop();
  }, [active, video, client, status, clock, stage]);

  return { stats, events, activeThreats, error, reset };
}
```

- [ ] **Step 7: Typecheck, lint, full tests**

Run: `cd /d/Projects/WeaponShield-AI/web && npm test && npm run typecheck && npm run lint`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/src/features/stage web/src/features/detection
git commit -m "feat(web): stage with viewfinder overlay, event log, FPS meter and video detection hook"
```

---

### Task 4: Alerts

**Files:**
- Create: `web/src/features/alerts/alert-gate.ts`, `web/src/features/alerts/tone.ts`, `web/src/features/alerts/use-threat-alerts.ts`, `web/src/features/alerts/AlertBanner.tsx`
- Test: `web/src/features/alerts/alert-gate.test.ts`, `web/src/features/alerts/AlertBanner.test.tsx`

**Interfaces:**
- Consumes: `ThreatEvent` (Plan 1)
- Produces:
  - `class AlertGate { constructor(cooldownMs?: number); shouldFire(now: number): boolean; reset(): void }` (default 10 000 ms)
  - `playAlertTone(): void` (Web Audio, two short tones; silently no-ops if audio is unavailable)
  - `useThreatAlerts(active: readonly ThreatEvent[], opts: { sound: boolean; notify: boolean }): { banner: ThreatEvent | null }` — fires sound/notification for events not seen before, through `AlertGate`; `banner` is the highest-confidence active event
  - `requestNotificationPermission(): Promise<boolean>`
  - `<AlertBanner event={ThreatEvent | null} armed={boolean} />` — `role="alert"`; nothing rendered when `event` is null

- [ ] **Step 1: Write the failing tests**

`web/src/features/alerts/alert-gate.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { AlertGate } from './alert-gate';

describe('AlertGate', () => {
  it('fires at most once per cooldown window', () => {
    const g = new AlertGate(10_000);
    expect(g.shouldFire(0)).toBe(true);
    expect(g.shouldFire(9_999)).toBe(false);
    expect(g.shouldFire(10_000)).toBe(true);
  });
  it('reset re-arms immediately', () => {
    const g = new AlertGate(10_000);
    g.shouldFire(0);
    g.reset();
    expect(g.shouldFire(1)).toBe(true);
  });
});
```

`web/src/features/alerts/AlertBanner.test.tsx`:
```tsx
// @vitest-environment jsdom
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { AlertBanner } from './AlertBanner';

const event = { id: 3, weaponTrackId: 9, cls: 'knife' as const, personTrackId: 2, startMs: 0, endMs: 0, maxConf: 0.87, ended: false };

describe('AlertBanner', () => {
  it('renders nothing without an event', () => {
    const { container } = render(<AlertBanner event={null} armed={false} />);
    expect(container).toBeEmptyDOMElement();
  });
  it('announces an armed person with the weapon and confidence', () => {
    render(<AlertBanner event={event} armed />);
    expect(screen.getByRole('alert')).toHaveTextContent(/armed person/i);
    expect(screen.getByRole('alert')).toHaveTextContent(/knife/i);
    expect(screen.getByRole('alert')).toHaveTextContent('87%');
  });
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/features/alerts`
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement**

`web/src/features/alerts/alert-gate.ts`:
```ts
/** Global rate limit for alert side effects (sound, notifications). */
export class AlertGate {
  private last: number | null = null;

  constructor(private readonly cooldownMs = 10_000) {}

  shouldFire(now: number): boolean {
    if (this.last !== null && now - this.last < this.cooldownMs) return false;
    this.last = now;
    return true;
  }

  reset(): void {
    this.last = null;
  }
}
```

`web/src/features/alerts/tone.ts`:
```ts
let ctx: AudioContext | null = null;

export function playAlertTone(): void {
  try {
    ctx ??= new AudioContext();
    const t0 = ctx.currentTime;
    for (const [i, freq] of [880, 660].entries()) {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'square';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, t0 + i * 0.18);
      gain.gain.exponentialRampToValueAtTime(0.08, t0 + i * 0.18 + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + i * 0.18 + 0.16);
      osc.connect(gain).connect(ctx.destination);
      osc.start(t0 + i * 0.18);
      osc.stop(t0 + i * 0.18 + 0.17);
    }
  } catch {
    // audio unavailable (autoplay policy or no device): the banner still shows
  }
}
```

`web/src/features/alerts/use-threat-alerts.ts`:
```ts
import { useEffect, useMemo, useRef } from 'react';
import type { ThreatEvent } from '@/core/threat';
import { AlertGate } from './alert-gate';
import { playAlertTone } from './tone';

export async function requestNotificationPermission(): Promise<boolean> {
  if (typeof Notification === 'undefined') return false;
  if (Notification.permission === 'granted') return true;
  if (Notification.permission === 'denied') return false;
  return (await Notification.requestPermission()) === 'granted';
}

export function useThreatAlerts(active: readonly ThreatEvent[], opts: { sound: boolean; notify: boolean }) {
  const seen = useRef(new Set<number>());
  const gate = useRef(new AlertGate());

  useEffect(() => {
    const fresh = active.filter((e) => !seen.current.has(e.id));
    if (fresh.length === 0) return;
    for (const e of fresh) seen.current.add(e.id);
    if (!gate.current.shouldFire(performance.now())) return;
    if (opts.sound) playAlertTone();
    if (opts.notify && typeof Notification !== 'undefined' && Notification.permission === 'granted') {
      const e = fresh[0]!;
      new Notification('WeaponShield alert', { body: `${e.cls} detected (${Math.round(e.maxConf * 100)}%)`, tag: `ws-${e.id}` });
    }
  }, [active, opts.sound, opts.notify]);

  const banner = useMemo(() => active.reduce<ThreatEvent | null>((best, e) => (!best || e.maxConf > best.maxConf ? e : best), null), [active]);
  return { banner };
}
```

`web/src/features/alerts/AlertBanner.tsx`:
```tsx
import { ShieldAlert } from 'lucide-react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import type { ThreatEvent } from '@/core/threat';

export function AlertBanner({ event, armed }: { event: ThreatEvent | null; armed: boolean }) {
  const reduce = useReducedMotion();
  return (
    <AnimatePresence>
      {event && (
        <motion.div
          key="alert"
          role="alert"
          initial={reduce ? false : { opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={reduce ? { opacity: 0 } : { opacity: 0, y: -8 }}
          className="flex items-center gap-3 rounded-[var(--radius-card)] border border-weapon/60 bg-weapon/15 px-4 py-3 text-sm text-fg"
        >
          <ShieldAlert className="size-5 shrink-0 text-weapon" aria-hidden="true" />
          <span className="font-semibold">{armed ? 'Armed person detected' : 'Weapon detected'}</span>
          <span className="font-mono uppercase text-weapon">{event.cls}</span>
          <span className="ml-auto font-mono tabular text-muted">{Math.round(event.maxConf * 100)}%</span>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
```

- [ ] **Step 4: Run tests and checks**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/features/alerts && npm test && npm run typecheck && npm run lint`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/src/features/alerts
git commit -m "feat(web): alert banner, two-tone sound and notifications with a 10 s global cooldown"
```

---

### Task 5: Demo clip assets

**Files:**
- Create: `web/demo.json`, `web/src/features/demo/demo-clips.ts`, `web/src/features/demo/DemoGallery.tsx`
- Modify: `web/scripts/prepare-assets.mjs`, `.gitignore` (add `web/public/demo/`)
- Test: `web/src/features/demo/demo-clips.test.ts`

**Interfaces:**
- Produces:
  - `web/demo.json` = `{ "clips": [{ "id", "title", "file", "url", "sha256", "author", "sourceUrl", "license", "shows": ClassKey[] }] }` (content in Step 2); files are served at `/demo/<file>`
  - `interface DemoClip { id: string; title: string; src: string; author: string; sourceUrl: string; license: string; shows: readonly ClassKey[] }`
  - `DEMO_CLIPS: readonly DemoClip[]`, `heroClip(): DemoClip` (first clip whose `shows` includes `'person'`)
  - `<DemoGallery onPick={(clip: DemoClip) => void} selectedId?: string />`

- [ ] **Step 1: Clips (already done by the controller)**

Four Pexels clips (H.264, 1280x720, 11–13 s, 1–4 MB each) were checked frame by frame and published as the pre-release `demo-v1`. Nothing to do in this step.

- [ ] **Step 2: Write `web/demo.json`**

```json
{
  "clips": [
    {
      "id": "man-aiming-pistol",
      "title": "Man aiming a pistol (shooting range)",
      "file": "man-aiming-pistol.mp4",
      "url": "https://github.com/Halok600/WeaponShield-AI/releases/download/demo-v1/man-aiming-pistol.mp4",
      "sha256": "acd59bc00f76332beb73d0805c7bb78655a03fa6f58332fbe4a2b2c10b35106d",
      "author": "Tima Miroshnichenko",
      "sourceUrl": "https://www.pexels.com/video/man-aiming-a-gun-6090906/",
      "license": "Pexels License",
      "shows": ["person", "pistol"]
    },
    {
      "id": "rifle-at-range",
      "title": "Man holding a rifle (shooting range)",
      "file": "rifle-at-range.mp4",
      "url": "https://github.com/Halok600/WeaponShield-AI/releases/download/demo-v1/rifle-at-range.mp4",
      "sha256": "18f246641721d4202b4c6a868d2592e2717872738c6dadf9c57995343ecd9121",
      "author": "Tima Miroshnichenko",
      "sourceUrl": "https://www.pexels.com/video/man-holding-a-rifle-standing-at-a-gun-range-6092114/",
      "license": "Pexels License",
      "shows": ["person", "rifle"]
    },
    {
      "id": "people-in-hallway",
      "title": "Crowded underground hallway",
      "file": "people-in-hallway.mp4",
      "url": "https://github.com/Halok600/WeaponShield-AI/releases/download/demo-v1/people-in-hallway.mp4",
      "sha256": "810f15f4910be496162da7874d90589dd0b163f1b97d9bd2297c3bc63182b41b",
      "author": "Jessica Politi",
      "sourceUrl": "https://www.pexels.com/video/people-in-a-hallway-992599/",
      "license": "Pexels License",
      "shows": ["person"]
    },
    {
      "id": "kitchen-knife",
      "title": "Slicing with a kitchen knife",
      "file": "kitchen-knife.mp4",
      "url": "https://github.com/Halok600/WeaponShield-AI/releases/download/demo-v1/kitchen-knife.mp4",
      "sha256": "3fc3de20947ac079fd9157b0794a4adfd7a5c8aa5ee9ddf4e4e97978cd4a14d8",
      "author": "Rachel Claire",
      "sourceUrl": "https://www.pexels.com/video/a-person-slicing-using-a-kitchen-knife-8714035/",
      "license": "Pexels License",
      "shows": ["knife"]
    }
  ]
}
```

- [ ] **Step 3: Extend `web/scripts/prepare-assets.mjs`**

Replace the `fetchModels` function with a generic downloader and call it for both manifests:
```js
async function fetchVerified(entries, dir) {
  await mkdir(dir, { recursive: true });
  for (const m of entries) {
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

const readJson = async (f) => JSON.parse(await readFile(join(web, f), 'utf8'));

await Promise.all([
  readJson('models.json').then((m) => fetchVerified(m.models, join(web, 'public', 'models'))),
  readJson('demo.json').then((d) => fetchVerified(d.clips, join(web, 'public', 'demo'))),
  copyOrtWasm(),
]);
console.log('prepare-assets: ok');
```
(Remove the old `fetchModels` and the old `await Promise.all(...)` line.) Add `web/public/demo/` to `.gitignore` under `# Node`.

- [ ] **Step 4: Write the failing test**

`web/src/features/demo/demo-clips.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { DEMO_CLIPS, heroClip } from './demo-clips';

describe('demo clips', () => {
  it('has attributed, same-origin clips', () => {
    expect(DEMO_CLIPS.length).toBeGreaterThanOrEqual(3);
    for (const c of DEMO_CLIPS) {
      expect(c.src).toMatch(/^\/demo\/[\w-]+\.mp4$/);
      expect(c.author.length).toBeGreaterThan(0);
      expect(c.sourceUrl).toMatch(/^https:\/\//);
      expect(c.license.length).toBeGreaterThan(0);
    }
  });
  it('picks a clip with people for the hero', () => {
    expect(heroClip().shows).toContain('person');
  });
});
```

- [ ] **Step 5: Implement**

`web/src/features/demo/demo-clips.ts`:
```ts
import type { ClassKey } from '@/core/types';
import manifest from '../../../demo.json';

export interface DemoClip {
  id: string;
  title: string;
  src: string;
  author: string;
  sourceUrl: string;
  license: string;
  shows: readonly ClassKey[];
}

export const DEMO_CLIPS: readonly DemoClip[] = manifest.clips.map((c) => ({
  id: c.id,
  title: c.title,
  src: `/demo/${c.file}`,
  author: c.author,
  sourceUrl: c.sourceUrl,
  license: c.license,
  shows: c.shows as ClassKey[],
}));

export function heroClip(): DemoClip {
  const clip = DEMO_CLIPS.find((c) => c.shows.includes('person'));
  if (!clip) throw new Error('demo.json needs at least one clip that shows a person');
  return clip;
}
```

`web/src/features/demo/DemoGallery.tsx`:
```tsx
import { Play } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/cn';
import { DEMO_CLIPS, type DemoClip } from './demo-clips';

export function DemoGallery({ onPick, selectedId }: { onPick: (clip: DemoClip) => void; selectedId?: string }) {
  return (
    <ul className="grid gap-3 sm:grid-cols-2">
      {DEMO_CLIPS.map((c) => (
        <li key={c.id}>
          <button
            type="button"
            onClick={() => onPick(c)}
            aria-pressed={c.id === selectedId}
            className={cn(
              'flex w-full items-start gap-3 rounded-lg border border-border bg-surface-2 p-3 text-left transition-colors hover:border-accent/50',
              c.id === selectedId && 'border-accent',
            )}
          >
            <span className="mt-0.5 rounded-md bg-accent/15 p-1.5 text-accent">
              <Play className="size-4" aria-hidden="true" />
            </span>
            <span className="min-w-0">
              <span className="block text-sm font-medium text-fg">{c.title}</span>
              <span className="mt-1 flex flex-wrap gap-1">
                {c.shows.map((s) => (
                  <Badge key={s} tone={s === 'person' ? 'person' : 'weapon'}>{s}</Badge>
                ))}
              </span>
              <span className="mt-1 block truncate text-xs text-muted">
                {c.author} · {c.license}
              </span>
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}
```

- [ ] **Step 6: Run tests and checks; verify the download**

```bash
cd /d/Projects/WeaponShield-AI/web
node scripts/prepare-assets.mjs && ls public/demo
npx vitest run src/features/demo && npm test && npm run typecheck && npm run lint
```
Expected: clips downloaded and verified; tests pass.

- [ ] **Step 7: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/demo.json web/scripts/prepare-assets.mjs web/src/features/demo .gitignore
git commit -m "feat(web): attributed demo clips downloaded and verified at build time, demo gallery"
```

---

### Task 6: Console — shell, model gate, side panel and Live mode

**Files:**
- Replace: `web/src/features/console/ConsolePage.tsx`
- Create: `web/src/features/console/{ModelGate,BackendBadge,StatsPanel,SensitivityControl,EventTimeline,AlertSettings,LiveMode,PanelLayout}.tsx`, `web/src/features/console/use-sensitivity.ts`
- Test: `web/src/features/console/EventTimeline.test.tsx`, `web/src/features/console/ModelGate.test.tsx`

**Interfaces:**
- Consumes: Tasks 1–5; `PRESETS`, `SensitivityName` (Plan 1)
- Produces:
  - `useSensitivity(): [SensitivityName, (s: SensitivityName) => void]` (persisted in localStorage key `ws-sensitivity`, default `'balanced'`, storage errors ignored)
  - `<ModelGate>{children}</ModelGate>` — calls `ensureReady()` on mount; renders children only when `status === 'ready'`; otherwise a card with progress (`role=progressbar`), or the error with a "Retry" button; test id `model-status` shows the status text
  - `<BackendBadge />` — test id `backend`, text `WebGPU` or `WASM`; WASM shows a tooltip explaining the fallback
  - `<StatsPanel stats={LiveStats} />` — test ids `stat-fps`, `stat-ms`, `stat-persons`, `stat-weapons`, `stat-armed`, `stat-frames`
  - `<SensitivityControl value onChange />`
  - `<EventTimeline events={EventRecord[]} onSelect?={(e: EventRecord) => void} onExport?={(kind: 'csv'|'json') => void} />` — empty state text "No threats detected yet"; each item is a button when `onSelect` is set
  - `<AlertSettings sound notify onSound onNotify />`
  - `<PanelLayout stage={ReactNode} panel={ReactNode} />` — stage left / panel right on `lg`, stacked below
  - `<LiveMode />` with test ids `camera-toggle`, `camera-select`; keyboard: Space toggles the camera, `S` saves a snapshot PNG
  - Console route `/console?mode=live|video|image` (default `live`)

- [ ] **Step 1: Write the failing tests**

`web/src/features/console/EventTimeline.test.tsx`:
```tsx
// @vitest-environment jsdom
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import type { EventRecord } from '@/features/detection/events';
import { EventTimeline } from './EventTimeline';

const rec = (id: number, over: Partial<EventRecord> = {}): EventRecord => ({
  id, weaponTrackId: id, cls: 'pistol', personTrackId: 4, startMs: 61_000, endMs: 64_500, maxConf: 0.91, ended: true, snapshot: null, ...over,
});

describe('EventTimeline', () => {
  it('shows an empty state', () => {
    render(<EventTimeline events={[]} />);
    expect(screen.getByText(/no threats detected yet/i)).toBeInTheDocument();
  });

  it('lists events with time, class, holder and confidence', () => {
    render(<EventTimeline events={[rec(1), rec(2, { cls: 'knife', personTrackId: null, ended: false })]} />);
    const items = screen.getAllByRole('listitem');
    expect(items[0]).toHaveTextContent('01:01');
    expect(items[0]).toHaveTextContent(/pistol/i);
    expect(items[0]).toHaveTextContent('#4');
    expect(items[0]).toHaveTextContent('91%');
    expect(items[1]).toHaveTextContent(/live/i);
  });

  it('calls onSelect and onExport', async () => {
    const onSelect = vi.fn();
    const onExport = vi.fn();
    render(<EventTimeline events={[rec(7)]} onSelect={onSelect} onExport={onExport} />);
    await userEvent.click(screen.getByRole('button', { name: /pistol/i }));
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ id: 7 }));
    await userEvent.click(screen.getByRole('button', { name: /csv/i }));
    expect(onExport).toHaveBeenCalledWith('csv');
  });
});
```

`web/src/features/console/ModelGate.test.tsx`:
```tsx
// @vitest-environment jsdom
import { act, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { DetectorProvider } from '@/app/detector-context';
import type { DetectorClient } from '@/engine/detector-client';
import { ModelGate } from './ModelGate';

const client = (p: () => Promise<{ backend: 'wasm'; loadMs: number }>) =>
  ({ onProgress: null, init: vi.fn(p), dispose: vi.fn() }) as unknown as DetectorClient;

describe('ModelGate', () => {
  it('shows progress, then the children', async () => {
    let resolve!: (v: { backend: 'wasm'; loadMs: number }) => void;
    render(
      <DetectorProvider createClient={() => client(() => new Promise((r) => (resolve = r)))}>
        <ModelGate>
          <p>console body</p>
        </ModelGate>
      </DetectorProvider>,
    );
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
    expect(screen.queryByText('console body')).not.toBeInTheDocument();
    await act(async () => resolve({ backend: 'wasm', loadMs: 5 }));
    expect(screen.getByText('console body')).toBeInTheDocument();
  });

  it('offers retry on failure', async () => {
    render(
      <DetectorProvider createClient={() => client(() => Promise.reject(new Error('HTTP 500')))}>
        <ModelGate>
          <p>body</p>
        </ModelGate>
      </DetectorProvider>,
    );
    await act(async () => {});
    expect(screen.getByText(/HTTP 500/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/features/console`
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement the panel components**

`web/src/features/console/use-sensitivity.ts`:
```ts
import { useState } from 'react';
import type { SensitivityName } from '@/core/presets';

const KEY = 'ws-sensitivity';
const VALID: readonly SensitivityName[] = ['low', 'balanced', 'high'];

function read(): SensitivityName {
  try {
    const v = localStorage.getItem(KEY);
    return VALID.includes(v as SensitivityName) ? (v as SensitivityName) : 'balanced';
  } catch {
    return 'balanced';
  }
}

export function useSensitivity(): [SensitivityName, (s: SensitivityName) => void] {
  const [value, setValue] = useState<SensitivityName>(read);
  return [
    value,
    (s) => {
      setValue(s);
      try {
        localStorage.setItem(KEY, s);
      } catch {
        // storage unavailable: keep the choice for this session only
      }
    },
  ];
}
```

`web/src/features/console/ModelGate.tsx`:
```tsx
import { AlertTriangle, Cpu } from 'lucide-react';
import { type ReactNode, useEffect } from 'react';
import { useDetector } from '@/app/detector-context';
import { Button } from '@/components/ui/button';
import { Card, CardBody } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';

export function ModelGate({ children }: { children: ReactNode }) {
  const { status, progress, error, ensureReady, retry } = useDetector();
  useEffect(() => ensureReady(), [ensureReady]);
  if (status === 'ready') return <>{children}</>;
  return (
    <Card className="mx-auto max-w-lg">
      <CardBody className="space-y-4 py-8 text-center">
        {status === 'error' ? (
          <>
            <AlertTriangle className="mx-auto size-8 text-weapon" aria-hidden="true" />
            <p data-testid="model-status" className="font-medium">The detection model failed to load</p>
            <p className="font-mono text-xs text-muted">{error}</p>
            <Button onClick={retry}>Retry</Button>
          </>
        ) : (
          <>
            <Cpu className="mx-auto size-8 text-accent" aria-hidden="true" />
            <p data-testid="model-status" className="font-medium">Loading the detection model…</p>
            <Progress value={progress} label="Model download progress" />
            <p className="text-xs text-muted">About 10 MB, downloaded once and cached. It runs entirely on this device.</p>
          </>
        )}
      </CardBody>
    </Card>
  );
}
```

`web/src/features/console/BackendBadge.tsx`:
```tsx
import { useDetector } from '@/app/detector-context';
import { Badge } from '@/components/ui/badge';
import { Tooltip } from '@/components/ui/tooltip';

export function BackendBadge() {
  const { backend } = useDetector();
  if (!backend) return null;
  if (backend === 'webgpu') {
    return <Badge tone="success" data-testid="backend">WebGPU</Badge>;
  }
  return (
    <Tooltip content="WebGPU isn’t available in this browser, so the model runs on the CPU with WebAssembly. Chrome or Edge with WebGPU is several times faster.">
      <Badge tone="armed" data-testid="backend" tabIndex={0}>WASM</Badge>
    </Tooltip>
  );
}
```

`web/src/features/console/StatsPanel.tsx`:
```tsx
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/card';
import { Stat } from '@/components/ui/stat';
import type { LiveStats } from '@/features/detection/use-video-detection';
import { BackendBadge } from './BackendBadge';

export function StatsPanel({ stats }: { stats: LiveStats }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Telemetry</CardTitle>
        <BackendBadge />
      </CardHeader>
      <CardBody className="grid grid-cols-3 gap-4">
        <Stat label="People" value={stats.persons} testId="stat-persons" />
        <Stat label="Weapons" value={stats.weapons} testId="stat-weapons" />
        <Stat label="Armed" value={stats.armed} testId="stat-armed" />
        <Stat label="FPS" value={stats.fps.toFixed(1)} testId="stat-fps" />
        <Stat label="Infer" value={stats.inferMs.toFixed(0)} unit="ms" testId="stat-ms" />
        <Stat label="Frames" value={stats.frames} testId="stat-frames" />
      </CardBody>
    </Card>
  );
}
```

`web/src/features/console/SensitivityControl.tsx`:
```tsx
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/card';
import { Segmented } from '@/components/ui/segmented';
import type { SensitivityName } from '@/core/presets';

const OPTIONS = [
  { value: 'low', label: 'Low' },
  { value: 'balanced', label: 'Balanced' },
  { value: 'high', label: 'High' },
] as const;

const HELP: Record<SensitivityName, string> = {
  low: 'Fewer alerts. Needs higher confidence and longer confirmation.',
  balanced: 'Recommended trade-off between missed weapons and false alarms.',
  high: 'Catches more, including partly hidden weapons, with more false alarms.',
};

export function SensitivityControl({ value, onChange }: { value: SensitivityName; onChange: (v: SensitivityName) => void }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Sensitivity</CardTitle>
      </CardHeader>
      <CardBody className="space-y-2">
        <Segmented value={value} onValueChange={onChange} options={OPTIONS} ariaLabel="Detection sensitivity" />
        <p className="text-xs text-muted">{HELP[value]}</p>
      </CardBody>
    </Card>
  );
}
```

`web/src/features/console/AlertSettings.tsx`:
```tsx
import { Bell, Volume2 } from 'lucide-react';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/card';
import { requestNotificationPermission } from '@/features/alerts/use-threat-alerts';

function Toggle({ checked, onChange, label, icon }: { checked: boolean; onChange: (v: boolean) => void; label: string; icon: React.ReactNode }) {
  return (
    <label className="flex cursor-pointer items-center gap-3 text-sm">
      <span className="text-muted [&_svg]:size-4">{icon}</span>
      <span className="flex-1">{label}</span>
      <input type="checkbox" className="size-4 accent-[var(--ws-accent)]" checked={checked} onChange={(e) => onChange(e.target.checked)} />
    </label>
  );
}

export function AlertSettings({ sound, notify, onSound, onNotify }: { sound: boolean; notify: boolean; onSound: (v: boolean) => void; onNotify: (v: boolean) => void }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Alerts</CardTitle>
      </CardHeader>
      <CardBody className="space-y-3">
        <Toggle checked={sound} onChange={onSound} label="Alarm sound" icon={<Volume2 />} />
        <Toggle
          checked={notify}
          onChange={async (v) => onNotify(v ? await requestNotificationPermission() : false)}
          label="Browser notifications"
          icon={<Bell />}
        />
      </CardBody>
    </Card>
  );
}
```

`web/src/features/console/EventTimeline.tsx`:
```tsx
import { Download } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/card';
import type { EventRecord } from '@/features/detection/events';

const mmss = (ms: number) => {
  const t = Math.max(0, Math.floor(ms / 1000));
  return `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`;
};

export function EventTimeline({
  events,
  onSelect,
  onExport,
  title = 'Event timeline',
}: {
  events: readonly EventRecord[];
  onSelect?: (e: EventRecord) => void;
  onExport?: (kind: 'csv' | 'json') => void;
  title?: string;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {onExport && events.length > 0 && (
          <div className="flex gap-1">
            <Button variant="ghost" size="sm" onClick={() => onExport('csv')}>
              <Download /> CSV
            </Button>
            <Button variant="ghost" size="sm" onClick={() => onExport('json')}>
              <Download /> JSON
            </Button>
          </div>
        )}
      </CardHeader>
      <CardBody className="max-h-[420px] overflow-y-auto p-2">
        {events.length === 0 ? (
          <p className="px-2 py-6 text-center text-sm text-muted">No threats detected yet</p>
        ) : (
          <ul className="space-y-2">
            {events.map((e) => {
              const body = (
                <>
                  {e.snapshot ? (
                    <img src={e.snapshot} alt="" className="h-14 w-24 shrink-0 rounded object-cover" />
                  ) : (
                    <span className="h-14 w-24 shrink-0 rounded bg-surface-2" aria-hidden="true" />
                  )}
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-2">
                      <Badge tone={e.personTrackId !== null ? 'armed' : 'weapon'}>{e.cls}</Badge>
                      {!e.ended && <Badge tone="weapon">Live</Badge>}
                    </span>
                    <span className="mt-1 block font-mono text-xs text-muted">
                      {mmss(e.startMs)} · {((e.endMs - e.startMs) / 1000).toFixed(1)}s
                      {e.personTrackId !== null && ` · held by #${e.personTrackId}`}
                    </span>
                  </span>
                  <span className="font-mono text-sm tabular text-fg">{Math.round(e.maxConf * 100)}%</span>
                </>
              );
              return (
                <li key={e.id}>
                  {onSelect ? (
                    <button type="button" onClick={() => onSelect(e)} className="flex w-full items-center gap-3 rounded-lg p-2 text-left hover:bg-surface-2">
                      {body}
                    </button>
                  ) : (
                    <div className="flex items-center gap-3 rounded-lg p-2">{body}</div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </CardBody>
    </Card>
  );
}
```

`web/src/features/console/PanelLayout.tsx`:
```tsx
import type { ReactNode } from 'react';

export function PanelLayout({ stage, panel }: { stage: ReactNode; panel: ReactNode }) {
  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
      <div className="min-w-0 space-y-3">{stage}</div>
      <aside className="space-y-4">{panel}</aside>
    </div>
  );
}
```

- [ ] **Step 4: Implement Live mode**

`web/src/lib/download.ts`:
```ts
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const downloadText = (text: string, filename: string, type: string) => downloadBlob(new Blob([text], { type }), filename);

export function dataUrlToBlob(dataUrl: string): Blob {
  const [meta, data] = dataUrl.split(',') as [string, string];
  const type = /data:(.*?);/.exec(meta)?.[1] ?? 'application/octet-stream';
  const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
  return new Blob([bytes], { type });
}
```

`web/src/features/console/LiveMode.tsx`:
```tsx
import { Camera, CameraOff, FlipHorizontal2 } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router';
import { Button, buttonClasses } from '@/components/ui/button';
import { Kbd } from '@/components/ui/kbd';
import { PRESETS } from '@/core/presets';
import { AlertBanner } from '@/features/alerts/AlertBanner';
import { useThreatAlerts } from '@/features/alerts/use-threat-alerts';
import { eventsToCsv, eventsToJson } from '@/features/detection/events';
import { useVideoDetection } from '@/features/detection/use-video-detection';
import { Stage, type StageHandle } from '@/features/stage/Stage';
import { dataUrlToBlob, downloadBlob, downloadText } from '@/lib/download';
import { AlertSettings } from './AlertSettings';
import { EventTimeline } from './EventTimeline';
import { PanelLayout } from './PanelLayout';
import { SensitivityControl } from './SensitivityControl';
import { StatsPanel } from './StatsPanel';
import { useSensitivity } from './use-sensitivity';

type CamState = 'off' | 'starting' | 'on' | 'denied' | 'unavailable';

export function LiveMode() {
  const stage = useRef<StageHandle>(null);
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);
  const [cam, setCam] = useState<CamState>('off');
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string>('');
  const [mirrored, setMirrored] = useState(true);
  const [sensitivity, setSensitivity] = useSensitivity();
  const [sound, setSound] = useState(true);
  const [notify, setNotify] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);

  const det = useVideoDetection({ video: videoEl, active: cam === 'on', preset: PRESETS[sensitivity], clock: 'wall', stage });
  const { banner } = useThreatAlerts(det.activeThreats, { sound, notify });

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoEl) videoEl.srcObject = null;
    stage.current?.drawFrame(null);
    setCam('off');
  }, [videoEl]);

  const startCamera = useCallback(async () => {
    if (!videoEl) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      setCam('unavailable');
      return;
    }
    setCam('starting');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: deviceId ? { deviceId: { exact: deviceId } } : { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      videoEl.srcObject = stream;
      await videoEl.play();
      setDevices((await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput'));
      det.reset();
      setCam('on');
    } catch (e) {
      setCam(e instanceof DOMException && e.name === 'NotAllowedError' ? 'denied' : 'unavailable');
    }
  }, [videoEl, deviceId, det]);

  useEffect(() => () => streamRef.current?.getTracks().forEach((t) => t.stop()), []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.target instanceof HTMLButtonElement) return;
      if (e.code === 'Space') {
        e.preventDefault();
        if (cam === 'on') stopCamera();
        else void startCamera();
      } else if (e.key.toLowerCase() === 's' && cam === 'on') {
        const shot = stage.current?.snapshot(videoEl?.videoWidth || 1280);
        if (shot) downloadBlob(dataUrlToBlob(shot), `weaponshield-${Date.now()}.jpg`);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [cam, startCamera, stopCamera, videoEl]);

  const overlay =
    cam === 'denied' ? (
      <div className="max-w-sm rounded-lg bg-black/80 p-4 text-center text-sm text-white">
        <p className="font-medium">Camera access was blocked</p>
        <p className="mt-1 text-white/70">Allow camera access in your browser’s site settings, or try a demo clip instead.</p>
        <Link to="/console?mode=video" className={buttonClasses({ variant: 'secondary', size: 'sm', className: 'mt-3' })}>Open demo clips</Link>
      </div>
    ) : cam === 'unavailable' ? (
      <div className="max-w-sm rounded-lg bg-black/80 p-4 text-center text-sm text-white">
        <p className="font-medium">No camera found</p>
        <Link to="/console?mode=video" className={buttonClasses({ variant: 'secondary', size: 'sm', className: 'mt-3' })}>Open demo clips</Link>
      </div>
    ) : cam === 'off' ? (
      <Button data-testid="camera-start-overlay" size="lg" onClick={() => void startCamera()}>
        <Camera /> Start camera
      </Button>
    ) : null;

  return (
    <PanelLayout
      stage={
        <>
          <AlertBanner event={banner} armed={det.stats.armed > 0} />
          <Stage ref={stage} label="CAM-01 · LIVE" live={cam === 'on'} mirrored={mirrored} videoEl={() => videoEl} video={<video ref={setVideoEl} muted playsInline />}>
            {overlay}
          </Stage>
          <div className="flex flex-wrap items-center gap-2">
            <Button data-testid="camera-toggle" variant={cam === 'on' ? 'danger' : 'primary'} onClick={() => (cam === 'on' ? stopCamera() : void startCamera())}>
              {cam === 'on' ? <CameraOff /> : <Camera />} {cam === 'on' ? 'Stop camera' : 'Start camera'}
            </Button>
            <Button variant="secondary" onClick={() => setMirrored((m) => !m)} aria-pressed={mirrored}>
              <FlipHorizontal2 /> Mirror
            </Button>
            {devices.length > 1 && (
              <select
                data-testid="camera-select"
                aria-label="Camera"
                value={deviceId}
                onChange={(e) => setDeviceId(e.target.value)}
                className="h-10 rounded-lg border border-border bg-surface-2 px-3 text-sm"
              >
                {devices.map((d, i) => (
                  <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>
                ))}
              </select>
            )}
            <span className="ml-auto hidden text-xs text-muted sm:inline">
              <Kbd>Space</Kbd> start/stop · <Kbd>S</Kbd> snapshot
            </span>
          </div>
          {det.error && <p className="text-sm text-weapon">{det.error}</p>}
        </>
      }
      panel={
        <>
          <StatsPanel stats={det.stats} />
          <SensitivityControl value={sensitivity} onChange={setSensitivity} />
          <AlertSettings sound={sound} notify={notify} onSound={setSound} onNotify={setNotify} />
          <EventTimeline
            events={det.events}
            onExport={(k) =>
              k === 'csv'
                ? downloadText(eventsToCsv(det.events), 'weaponshield-events.csv', 'text/csv')
                : downloadText(eventsToJson(det.events), 'weaponshield-events.json', 'application/json')
            }
          />
        </>
      }
    />
  );
}
```

`web/src/features/console/ConsolePage.tsx`:
```tsx
import { Camera, Film, Image as ImageIcon } from 'lucide-react';
import { useSearchParams } from 'react-router';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LiveMode } from './LiveMode';
import { ModelGate } from './ModelGate';

const MODES = ['live', 'video', 'image'] as const;
type Mode = (typeof MODES)[number];

export function ConsolePage() {
  const [params, setParams] = useSearchParams();
  const mode: Mode = MODES.includes(params.get('mode') as Mode) ? (params.get('mode') as Mode) : 'live';
  return (
    <section className="py-8">
      <div className="mb-6">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.14em] text-accent">Detection console</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight sm:text-3xl">Watch for weapons in real time</h1>
          <p className="mt-1 text-sm text-muted">Everything runs on this device. No frame ever leaves your browser.</p>
        </div>
      </div>
      {/* One Tabs root so each trigger's aria-controls points at a rendered panel. */}
      <Tabs value={mode} onValueChange={(v) => setParams({ mode: v }, { replace: true })}>
        <TabsList aria-label="Input">
          <TabsTrigger value="live"><Camera /> Live</TabsTrigger>
          <TabsTrigger value="video"><Film /> Video</TabsTrigger>
          <TabsTrigger value="image"><ImageIcon /> Image</TabsTrigger>
        </TabsList>
        <TabsContent value="live"><ModelGate><LiveMode /></ModelGate></TabsContent>
        <TabsContent value="video"><ModelGate><p className="text-sm text-muted">Video mode arrives in the next task.</p></ModelGate></TabsContent>
        <TabsContent value="image"><ModelGate><p className="text-sm text-muted">Image mode arrives in the next task.</p></ModelGate></TabsContent>
      </Tabs>
    </section>
  );
}
```
(The two temporary `<p>` lines are replaced in Tasks 7 and 8.)

- [ ] **Step 5: Run tests and checks, and try it**

```bash
cd /d/Projects/WeaponShield-AI/web
npx vitest run src/features/console && npm test && npm run typecheck && npm run lint && npm run build
```
Expected: all pass. Then `npm run dev`, open `/console`, start the camera (real webcam): boxes labelled `#1 person` follow you; the telemetry FPS updates; the mirror toggle flips the image while labels stay readable.

- [ ] **Step 6: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/src/features/console web/src/lib/download.ts
git commit -m "feat(web): console shell with model gate, telemetry, sensitivity, alerts and live camera mode"
```

---

### Task 7: Console — Video mode with timeline seek and exports

**Files:**
- Create: `web/src/features/console/clip-recorder.ts`, `web/src/features/console/VideoMode.tsx`
- Modify: `web/src/features/console/ConsolePage.tsx` (video tab)
- Test: `web/src/features/console/clip-recorder.test.ts`

**Interfaces:**
- Consumes: Tasks 3–6, `DemoGallery`, `DEMO_CLIPS`
- Produces:
  - `pickWebmMime(isSupported: (t: string) => boolean): string | null` — first of `video/webm;codecs=vp9`, `video/webm;codecs=vp8`, `video/webm`
  - `recordCanvas(canvas: HTMLCanvasElement, fps?: number): { stop(): Promise<Blob> }` (throws `Error('Recording is not supported in this browser')` when no mime type is supported)
  - `<VideoMode />` with test ids `video-file`, `video-play`, `video-record`; demo gallery; timeline items seek the video to `startMs`

- [ ] **Step 1: Write the failing test**

`web/src/features/console/clip-recorder.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { pickWebmMime } from './clip-recorder';

describe('pickWebmMime', () => {
  it('prefers VP9, then VP8, then plain WebM', () => {
    expect(pickWebmMime(() => true)).toBe('video/webm;codecs=vp9');
    expect(pickWebmMime((t) => t !== 'video/webm;codecs=vp9')).toBe('video/webm;codecs=vp8');
    expect(pickWebmMime((t) => t === 'video/webm')).toBe('video/webm');
    expect(pickWebmMime(() => false)).toBeNull();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/features/console/clip-recorder.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `web/src/features/console/clip-recorder.ts`**

```ts
const CANDIDATES = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'] as const;

export function pickWebmMime(isSupported: (t: string) => boolean): string | null {
  return CANDIDATES.find((t) => isSupported(t)) ?? null;
}

export function recordCanvas(canvas: HTMLCanvasElement, fps = 25): { stop(): Promise<Blob> } {
  const mime = typeof MediaRecorder === 'undefined' ? null : pickWebmMime((t) => MediaRecorder.isTypeSupported(t));
  if (!mime) throw new Error('Recording is not supported in this browser');
  const chunks: Blob[] = [];
  const recorder = new MediaRecorder(canvas.captureStream(fps), { mimeType: mime, videoBitsPerSecond: 4_000_000 });
  recorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };
  recorder.start(250);
  return {
    stop: () =>
      new Promise<Blob>((resolve) => {
        recorder.onstop = () => resolve(new Blob(chunks, { type: 'video/webm' }));
        recorder.stop();
      }),
  };
}
```

- [ ] **Step 4: Implement `web/src/features/console/VideoMode.tsx`**

```tsx
import { Circle, Pause, Play, Square, Upload } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/card';
import { PRESETS } from '@/core/presets';
import { AlertBanner } from '@/features/alerts/AlertBanner';
import { useThreatAlerts } from '@/features/alerts/use-threat-alerts';
import { DemoGallery } from '@/features/demo/DemoGallery';
import { DEMO_CLIPS, type DemoClip } from '@/features/demo/demo-clips';
import { eventsToCsv, eventsToJson } from '@/features/detection/events';
import { useVideoDetection } from '@/features/detection/use-video-detection';
import { Stage, type StageHandle } from '@/features/stage/Stage';
import { downloadBlob, downloadText } from '@/lib/download';
import { AlertSettings } from './AlertSettings';
import { recordCanvas } from './clip-recorder';
import { EventTimeline } from './EventTimeline';
import { PanelLayout } from './PanelLayout';
import { SensitivityControl } from './SensitivityControl';
import { StatsPanel } from './StatsPanel';
import { useSensitivity } from './use-sensitivity';

interface Source {
  src: string;
  name: string;
  demoId?: string;
  objectUrl?: boolean;
}

export function VideoMode() {
  const stage = useRef<StageHandle>(null);
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);
  const [source, setSource] = useState<Source>(() => ({ src: DEMO_CLIPS[0]!.src, name: DEMO_CLIPS[0]!.title, demoId: DEMO_CLIPS[0]!.id }));
  const [playing, setPlaying] = useState(false);
  const [recording, setRecording] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [sensitivity, setSensitivity] = useSensitivity();
  const [sound, setSound] = useState(false);
  const [notify, setNotify] = useState(false);
  const recorder = useRef<{ stop(): Promise<Blob> } | null>(null);
  const compositeTimer = useRef<number | null>(null);

  const det = useVideoDetection({ video: videoEl, active: playing, preset: PRESETS[sensitivity], clock: 'media', stage });
  const { banner } = useThreatAlerts(det.activeThreats, { sound, notify });

  useEffect(() => () => {
    if (source.objectUrl) URL.revokeObjectURL(source.src);
  }, [source]);

  const load = (next: Source) => {
    setPlaying(false);
    setLoadError(null);
    det.reset();
    setSource(next);
  };

  const pickDemo = (c: DemoClip) => load({ src: c.src, name: c.title, demoId: c.id });

  const togglePlay = async () => {
    if (!videoEl) return;
    if (videoEl.paused) {
      await videoEl.play();
      setPlaying(true);
    } else {
      videoEl.pause();
      setPlaying(false);
    }
  };

  const stopRecording = async () => {
    if (compositeTimer.current !== null) cancelAnimationFrame(compositeTimer.current);
    compositeTimer.current = null;
    const r = recorder.current;
    recorder.current = null;
    setRecording(false);
    if (r) downloadBlob(await r.stop(), `weaponshield-${source.demoId ?? 'clip'}-annotated.webm`);
  };

  const startRecording = async () => {
    if (!videoEl) return;
    const canvas = document.createElement('canvas');
    try {
      stage.current?.composite(canvas);
      recorder.current = recordCanvas(canvas);
    } catch (e) {
      setLoadError(e instanceof Error ? e.message : String(e));
      return;
    }
    const loop = () => {
      stage.current?.composite(canvas);
      compositeTimer.current = requestAnimationFrame(loop);
    };
    loop();
    setRecording(true);
    det.reset();
    videoEl.currentTime = 0;
    await videoEl.play();
    setPlaying(true);
  };

  return (
    <PanelLayout
      stage={
        <>
          <AlertBanner event={banner} armed={det.stats.armed > 0} />
          <Stage
            ref={stage}
            label={`FILE · ${source.name}`}
            live={playing}
            videoEl={() => videoEl}
            video={
              <video
                ref={setVideoEl}
                key={source.src}
                src={source.src}
                muted
                playsInline
                preload="auto"
                onEnded={() => {
                  setPlaying(false);
                  if (recording) void stopRecording();
                }}
                onError={() => setLoadError('This video can’t be decoded by your browser. Try an MP4 (H.264) file.')}
              />
            }
          />
          <div className="flex flex-wrap items-center gap-2">
            <Button data-testid="video-play" onClick={() => void togglePlay()}>
              {playing ? <Pause /> : <Play />} {playing ? 'Pause' : 'Play & detect'}
            </Button>
            <label className="inline-flex h-10 cursor-pointer items-center gap-2 rounded-lg border border-border bg-surface-2 px-4 text-sm hover:bg-surface-2/70">
              <Upload className="size-4" /> Open video
              <input
                data-testid="video-file"
                type="file"
                accept="video/*"
                className="sr-only"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) load({ src: URL.createObjectURL(f), name: f.name, objectUrl: true });
                }}
              />
            </label>
            <Button data-testid="video-record" variant={recording ? 'danger' : 'secondary'} onClick={() => void (recording ? stopRecording() : startRecording())}>
              {recording ? <Square /> : <Circle />} {recording ? 'Stop recording' : 'Export annotated clip'}
            </Button>
          </div>
          {(loadError || det.error) && <p className="text-sm text-weapon">{loadError ?? det.error}</p>}
          <Card>
            <CardHeader>
              <CardTitle>Demo clips</CardTitle>
            </CardHeader>
            <CardBody>
              <DemoGallery onPick={pickDemo} selectedId={source.demoId} />
            </CardBody>
          </Card>
        </>
      }
      panel={
        <>
          <StatsPanel stats={det.stats} />
          <SensitivityControl value={sensitivity} onChange={setSensitivity} />
          <AlertSettings sound={sound} notify={notify} onSound={setSound} onNotify={setNotify} />
          <EventTimeline
            events={det.events}
            onSelect={(e) => {
              if (videoEl) videoEl.currentTime = e.startMs / 1000;
            }}
            onExport={(k) =>
              k === 'csv'
                ? downloadText(eventsToCsv(det.events), 'weaponshield-events.csv', 'text/csv')
                : downloadText(eventsToJson(det.events), 'weaponshield-events.json', 'application/json')
            }
          />
        </>
      }
    />
  );
}
```

In `ConsolePage.tsx`, import `VideoMode` and replace the temporary video `<p>` (inside its `ModelGate`) with `<VideoMode />`.

- [ ] **Step 5: Run tests and checks, and try it**

```bash
cd /d/Projects/WeaponShield-AI/web
npx vitest run src/features/console && npm test && npm run typecheck && npm run lint && npm run build
```
Expected: all pass. In `npm run dev` → `/console?mode=video`: the first demo clip plays with boxes and trails; clicking a timeline event seeks; "Export annotated clip" downloads a `.webm` that plays with boxes burned in.

- [ ] **Step 6: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/src/features/console
git commit -m "feat(web): video mode with demo gallery, timeline seek, CSV/JSON export and annotated WebM export"
```

---

### Task 8: Console — Image mode

**Files:**
- Create: `web/src/features/console/ImageMode.tsx`
- Modify: `web/src/features/console/ConsolePage.tsx` (image tab)

**Interfaces:**
- Consumes: `useDetector`, `PRESETS`, `filterForImage`, `buildDetectionPrimitives`, `paint`, `downloadBlob`, `CLASS_KEYS`
- Produces: `<ImageMode />` with test ids `image-input`, `image-drop`, `image-result`, `count-person`, `count-weapon`, `image-download`; supports drag-and-drop and file picker; uses `PRESETS[sensitivity].imageConf`

- [ ] **Step 1: Implement `web/src/features/console/ImageMode.tsx`**

```tsx
import { Download, ImageUp } from 'lucide-react';
import { useRef, useState } from 'react';
import { useDetector } from '@/app/detector-context';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/card';
import { filterForImage, PRESETS } from '@/core/presets';
import { type Detection, isWeapon } from '@/core/types';
import { downloadBlob } from '@/lib/download';
import { cn } from '@/lib/cn';
import { buildDetectionPrimitives, paint } from '@/render/overlay';
import { PanelLayout } from './PanelLayout';
import { SensitivityControl } from './SensitivityControl';
import { useSensitivity } from './use-sensitivity';

export function ImageMode() {
  const { client } = useDetector();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dets, setDets] = useState<Detection[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [inferMs, setInferMs] = useState(0);
  const [sensitivity, setSensitivity] = useSensitivity();

  async function analyse(file: File) {
    const canvas = canvasRef.current;
    if (!client || !canvas) return;
    if (!file.type.startsWith('image/')) {
      setError('Please choose an image file (JPEG, PNG or WebP).');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const shown = await createImageBitmap(file);
      const sent = await createImageBitmap(file);
      canvas.width = shown.width;
      canvas.height = shown.height;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(shown, 0, 0);
      shown.close();
      const r = await client.detect(sent, PRESETS[sensitivity].lowConf);
      const kept = filterForImage(r.detections, PRESETS[sensitivity].imageConf);
      const scale = Math.max(1, canvas.width / 1280);
      ctx.save();
      paint(ctx, buildDetectionPrimitives(kept), scale, scale);
      ctx.restore();
      setDets(kept);
      setInferMs(r.inferMs);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const persons = dets?.filter((d) => d.cls === 'person').length ?? 0;
  const weapons = dets?.filter((d) => isWeapon(d.cls)).length ?? 0;

  return (
    <PanelLayout
      stage={
        <>
          <label
            data-testid="image-drop"
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const f = e.dataTransfer.files[0];
              if (f) void analyse(f);
            }}
            className={cn(
              'relative flex min-h-64 cursor-pointer items-center justify-center overflow-hidden rounded-[var(--radius-card)] border border-dashed border-border bg-surface',
              dragging && 'border-accent bg-accent/5',
            )}
          >
            <canvas data-testid="image-result" ref={canvasRef} className={cn('max-h-[70vh] w-full object-contain', !dets && 'hidden')} />
            {!dets && (
              <span className="flex flex-col items-center gap-2 p-8 text-center text-sm text-muted">
                <ImageUp className="size-8 text-accent" aria-hidden="true" />
                <span className="font-medium text-fg">Drop an image or click to choose</span>
                <span>JPEG, PNG or WebP · analysed on this device</span>
              </span>
            )}
            <input
              data-testid="image-input"
              type="file"
              accept="image/*"
              className="sr-only"
              disabled={busy}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void analyse(f);
              }}
            />
          </label>
          {error && <p className="text-sm text-weapon">{error}</p>}
        </>
      }
      panel={
        <>
          <Card>
            <CardHeader>
              <CardTitle>Result</CardTitle>
              {dets && (
                <Button
                  data-testid="image-download"
                  variant="ghost"
                  size="sm"
                  onClick={() => canvasRef.current?.toBlob((b) => b && downloadBlob(b, 'weaponshield-annotated.png'), 'image/png')}
                >
                  <Download /> PNG
                </Button>
              )}
            </CardHeader>
            <CardBody className="space-y-3">
              <div className="grid grid-cols-3 gap-3 font-mono">
                <div><div className="text-[11px] uppercase tracking-wider text-muted">People</div><div data-testid="count-person" className="text-2xl font-semibold">{persons}</div></div>
                <div><div className="text-[11px] uppercase tracking-wider text-muted">Weapons</div><div data-testid="count-weapon" className="text-2xl font-semibold">{weapons}</div></div>
                <div><div className="text-[11px] uppercase tracking-wider text-muted">Time</div><div className="text-2xl font-semibold">{inferMs.toFixed(0)}<span className="text-xs text-muted"> ms</span></div></div>
              </div>
              {dets && dets.length > 0 && (
                <ul className="space-y-1 text-sm">
                  {dets.map((d, i) => (
                    <li key={i} className="flex items-center justify-between">
                      <Badge tone={d.cls === 'person' ? 'person' : 'weapon'}>{d.cls}</Badge>
                      <span className="font-mono tabular text-muted">{Math.round(d.conf * 100)}%</span>
                    </li>
                  ))}
                </ul>
              )}
              {dets && dets.length === 0 && <p className="text-sm text-muted">Nothing detected at this sensitivity.</p>}
            </CardBody>
          </Card>
          <SensitivityControl value={sensitivity} onChange={setSensitivity} />
        </>
      }
    />
  );
}
```

In `ConsolePage.tsx`, import `ImageMode` and replace the temporary image `<p>` (inside its `ModelGate`) with `<ImageMode />`.

- [ ] **Step 2: Checks and manual try**

```bash
cd /d/Projects/WeaponShield-AI/web
npm test && npm run typecheck && npm run lint && npm run build
```
Expected: pass. In dev: `/console?mode=image`, drop `fixtures/golden/images/bus.jpg` → at least 3 `person` boxes; PNG download works.

- [ ] **Step 3: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/src/features/console
git commit -m "feat(web): image mode with drag-and-drop, per-class results and annotated PNG download"
```

---

### Task 9: Landing page

**Files:**
- Replace: `web/src/features/landing/LandingPage.tsx`
- Create: `web/src/features/landing/HeroDemo.tsx`, `web/src/features/landing/HowItWorks.tsx`
- Create: `web/src/content/model-card.json`, `web/src/content/model-card.ts`
- Test: `web/src/content/model-card.test.ts`

**Interfaces:**
- Consumes: `useVideoDetection`, `Stage`, `heroClip`, `useDetector`, `PRESETS`
- Produces:
  - `web/src/content/model-card.json` — `{ "status": "pending" | "ready", "model": {...}, "test"?: {...}, ... }` (full schema in Step 1)
  - `type ModelCard`, `MODEL_CARD: ModelCard`, `weaponsMap50(): number | null`
  - `<HeroDemo onStats={(s: LiveStats) => void} />` — autoplaying, muted, looping hero clip with detection overlay once the model is ready; test id `hero-detections` (count of person + weapon tracks currently visible)
  - Landing sections: hero (headline, subhead, CTAs to `/console` and `/model`), metrics strip (test ids `metric-map`, `metric-fps`, `metric-backend`, `metric-uploaded`), How it works, privacy + edge-tool band

- [ ] **Step 1: Model card content and accessors**

`web/src/content/model-card.json` (pending state; Plan 5 replaces it with real results):
```json
{
  "status": "pending",
  "model": {
    "name": "Stock YOLO26n (COCO)",
    "file": "yolo26n-coco.onnx",
    "classes": ["person", "knife"],
    "note": "Development model used while the WeaponShield model trains. It knows people and knives from COCO, not pistols or rifles."
  },
  "evaluation": {
    "protocol": [
      "Held-out test split never used for training, model selection or threshold tuning.",
      "Near-duplicate images (perceptual hash) and frames from the same source video are kept in the same split.",
      "Per-class precision, recall, mAP50 and mAP50-95 on the test split.",
      "False-alarm rate on 500 hard-negative images of people holding phones, tools and other look-alike objects.",
      "Thresholds for Low, Balanced and High chosen on the validation split."
    ]
  }
}
```

`web/src/content/model-card.ts`:
```ts
import raw from './model-card.json';

export type ClassName = 'person' | 'pistol' | 'rifle' | 'knife';
export interface ClassMetrics {
  precision: number;
  recall: number;
  map50: number;
  map50_95: number;
}

export interface ModelCard {
  status: 'pending' | 'ready';
  model: { name: string; file: string; classes: string[]; note?: string; params?: string; sizeMb?: number };
  evaluation: { protocol: string[] };
  test?: { per_class: Partial<Record<ClassName, ClassMetrics>>; weapons: ClassMetrics; map50_all: number; map50_95_all: number };
  baseline?: { name: string; weapons_map50: number; note: string };
  thresholds?: Record<'low' | 'balanced' | 'high', Partial<Record<ClassName, number>>>;
  hardneg?: { images: number; fp_rate: Record<'low' | 'balanced' | 'high', number> };
  pr_curves?: Partial<Record<ClassName, [number, number][]>>;
  confusion?: { labels: string[]; matrix: number[][] };
  dataset?: { splits: Record<string, { images: number; instances: Record<ClassName, number> }>; sources: Record<string, number> };
  gallery?: { file: string; kind: 'miss' | 'false-alarm'; class: ClassName; conf?: number }[];
  browser_fps?: { webgpu?: number; wasm?: number; device: string };
}

export const MODEL_CARD = raw as ModelCard;

export function weaponsMap50(card: ModelCard = MODEL_CARD): number | null {
  return card.status === 'ready' && card.test ? card.test.weapons.map50 : null;
}
```

`web/src/content/model-card.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { MODEL_CARD, type ModelCard, weaponsMap50 } from './model-card';

describe('model card', () => {
  it('ships a valid card', () => {
    expect(['pending', 'ready']).toContain(MODEL_CARD.status);
    expect(MODEL_CARD.evaluation.protocol.length).toBeGreaterThan(0);
  });
  it('only reports a headline metric when results are ready', () => {
    const pending: ModelCard = { ...MODEL_CARD, status: 'pending' };
    expect(weaponsMap50(pending)).toBeNull();
    const ready: ModelCard = {
      ...MODEL_CARD,
      status: 'ready',
      test: { per_class: {}, weapons: { precision: 0.9, recall: 0.8, map50: 0.87, map50_95: 0.6 }, map50_all: 0.88, map50_95_all: 0.61 },
    };
    expect(weaponsMap50(ready)).toBe(0.87);
  });
});
```

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/content` — expected PASS after creating both files (write the test first, run it to see it fail with "cannot resolve ./model-card", then add the files).

- [ ] **Step 2: Implement `web/src/features/landing/HeroDemo.tsx`**

```tsx
import { useEffect, useRef, useState } from 'react';
import { useDetector } from '@/app/detector-context';
import { Progress } from '@/components/ui/progress';
import { PRESETS } from '@/core/presets';
import { heroClip } from '@/features/demo/demo-clips';
import { type LiveStats, useVideoDetection } from '@/features/detection/use-video-detection';
import { Stage, type StageHandle } from '@/features/stage/Stage';

export function HeroDemo({ onStats }: { onStats: (s: LiveStats) => void }) {
  const clip = heroClip();
  const stage = useRef<StageHandle>(null);
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const { status, progress, ensureReady } = useDetector();
  useEffect(() => ensureReady(), [ensureReady]);

  const det = useVideoDetection({ video: videoEl, active: playing && status === 'ready', preset: PRESETS.balanced, clock: 'media', stage });
  useEffect(() => onStats(det.stats), [det.stats, onStats]);

  return (
    <div className="relative">
      <Stage
        ref={stage}
        label="CAM-02 · DEMO"
        live={playing && status === 'ready'}
        videoEl={() => videoEl}
        video={
          <video
            ref={setVideoEl}
            src={clip.src}
            autoPlay
            muted
            loop
            playsInline
            preload="auto"
            onPlaying={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            aria-label={`Demo clip: ${clip.title}`}
          />
        }
      >
        {status !== 'ready' && (
          <div className="w-56 rounded-lg bg-black/70 p-3 text-center font-mono text-[11px] uppercase tracking-wider text-white">
            {status === 'error' ? 'Model failed to load' : `Loading model ${Math.round(progress * 100)}%`}
            {status !== 'error' && <Progress value={progress} label="Model download progress" className="mt-2" />}
          </div>
        )}
      </Stage>
      <span data-testid="hero-detections" className="sr-only">{det.stats.persons + det.stats.weapons}</span>
      <p className="mt-2 text-right text-[11px] text-muted">
        Clip: <a href={clip.sourceUrl} target="_blank" rel="noreferrer" className="underline hover:text-fg">{clip.author}</a> · {clip.license}
      </p>
    </div>
  );
}
```

- [ ] **Step 3: Implement `web/src/features/landing/HowItWorks.tsx`**

```tsx
import { BellRing, Crosshair, ScanEye } from 'lucide-react';
import { Card, CardBody } from '@/components/ui/card';

const STEPS = [
  { icon: ScanEye, title: 'Detect', body: 'A YOLO26 model finds people, pistols, rifles and knives in every frame, on your GPU with WebGPU or on the CPU with WebAssembly.' },
  { icon: Crosshair, title: 'Track', body: 'A Kalman-filter tracker gives each person and weapon a stable ID, so one noisy frame can’t raise an alarm on its own.' },
  { icon: BellRing, title: 'Alert', body: 'A weapon seen in 4 of the last 10 frames is confirmed. If it’s in someone’s hands they’re flagged ARMED, and you get a banner, a sound and a timeline entry.' },
] as const;

export function HowItWorks() {
  return (
    <div className="grid gap-4 md:grid-cols-3">
      {STEPS.map(({ icon: Icon, title, body }, i) => (
        <Card key={title}>
          <CardBody className="space-y-3">
            <div className="flex items-center gap-3">
              <span className="rounded-lg bg-accent/15 p-2 text-accent"><Icon className="size-5" aria-hidden="true" /></span>
              <span className="font-mono text-xs text-muted">0{i + 1}</span>
            </div>
            <h3 className="text-lg font-semibold tracking-tight">{title}</h3>
            <p className="text-sm leading-relaxed text-muted">{body}</p>
          </CardBody>
        </Card>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Implement `web/src/features/landing/LandingPage.tsx`**

```tsx
import { ArrowRight, Lock, Router } from 'lucide-react';
import { motion, useReducedMotion } from 'motion/react';
import { useCallback, useState } from 'react';
import { Link } from 'react-router';
import { useDetector } from '@/app/detector-context';
import { REPO_URL } from '@/components/layout/SiteHeader';
import { Badge } from '@/components/ui/badge';
import { buttonClasses } from '@/components/ui/button';
import { Card, CardBody } from '@/components/ui/card';
import { Stat } from '@/components/ui/stat';
import { weaponsMap50 } from '@/content/model-card';
import type { LiveStats } from '@/features/detection/use-video-detection';
import { HeroDemo } from './HeroDemo';
import { HowItWorks } from './HowItWorks';

export function LandingPage() {
  const reduce = useReducedMotion();
  const { backend } = useDetector();
  const [fps, setFps] = useState(0);
  const onStats = useCallback((s: LiveStats) => setFps(s.fps), []);
  const map = weaponsMap50();
  const rise = (delay: number) => (reduce ? {} : { initial: { opacity: 0, y: 12 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.5, delay } });

  return (
    <>
      <section className="relative -mx-4 overflow-hidden px-4 pb-12 pt-10 sm:pt-16">
        <div aria-hidden="true" className="bg-grid pointer-events-none absolute inset-0 [mask-image:radial-gradient(ellipse_at_top,black,transparent_70%)]" />
        <div className="relative grid items-center gap-10 lg:grid-cols-[1fr_1.15fr]">
          <div>
            <motion.div {...rise(0)}>
              <Badge tone="accent">Runs 100% in your browser</Badge>
            </motion.div>
            <motion.h1 {...rise(0.05)} className="mt-5 text-4xl font-semibold leading-[1.05] tracking-tight sm:text-5xl lg:text-6xl">
              Real-time weapon detection, <span className="text-accent">on your device.</span>
            </motion.h1>
            <motion.p {...rise(0.1)} className="mt-5 max-w-xl text-lg leading-relaxed text-muted">
              WeaponShield spots pistols, rifles and knives, tracks the people holding them and raises an alert, in real time, with nothing uploaded.
            </motion.p>
            <motion.div {...rise(0.15)} className="mt-8 flex flex-wrap gap-3">
              <Link to="/console" className={buttonClasses({ size: 'lg' })}>
                Open the console <ArrowRight />
              </Link>
              <Link to="/model" className={buttonClasses({ size: 'lg', variant: 'secondary' })}>See the model card</Link>
            </motion.div>
          </div>
          <motion.div {...rise(0.1)}>
            <HeroDemo onStats={onStats} />
          </motion.div>
        </div>
      </section>

      <section aria-label="Key numbers" className="grid grid-cols-2 gap-4 rounded-[var(--radius-card)] border border-border bg-surface p-5 md:grid-cols-4">
        <Stat label="Weapons mAP50" value={map === null ? 'Training' : map.toFixed(3)} hint={map === null ? 'Results publish when training ends' : 'Held-out test set'} testId="metric-map" />
        <Stat label="On this device" value={fps > 0 ? fps.toFixed(0) : '—'} unit="FPS" hint="Measured live in the demo" testId="metric-fps" />
        <Stat label="Running on" value={backend === 'webgpu' ? 'WebGPU' : backend === 'wasm' ? 'WASM' : '—'} hint="Picked automatically" testId="metric-backend" />
        <Stat label="Uploaded" value="0" unit="bytes" hint="Frames never leave the browser" testId="metric-uploaded" />
      </section>

      <section className="py-16">
        <p className="font-mono text-xs uppercase tracking-[0.14em] text-accent">How it works</p>
        <h2 className="mt-2 text-2xl font-semibold tracking-tight sm:text-3xl">Detect, track, alert</h2>
        <div className="mt-8">
          <HowItWorks />
        </div>
        <Link to="/architecture" className="mt-6 inline-flex items-center gap-1 text-sm text-accent hover:underline">
          Read the full pipeline <ArrowRight className="size-4" />
        </Link>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardBody className="space-y-2">
            <Lock className="size-5 text-accent" aria-hidden="true" />
            <h3 className="text-lg font-semibold">Private by design</h3>
            <p className="text-sm leading-relaxed text-muted">
              The model is downloaded once (about 10 MB) and runs in a Web Worker. Camera frames, videos and images are processed in memory on your device and never sent anywhere.
            </p>
          </CardBody>
        </Card>
        <Card>
          <CardBody className="space-y-2">
            <Router className="size-5 text-accent" aria-hidden="true" />
            <h3 className="text-lg font-semibold">Real CCTV cameras too</h3>
            <p className="text-sm leading-relaxed text-muted">
              Browsers can’t open RTSP streams, so the open-source edge tool runs the same model next to your cameras and sends Telegram alerts with a snapshot.
            </p>
            <a href={`${REPO_URL}/tree/v2/edge`} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-sm text-accent hover:underline">
              Edge tool on GitHub <ArrowRight className="size-4" />
            </a>
          </CardBody>
        </Card>
      </section>
    </>
  );
}
```

- [ ] **Step 5: Checks and manual look**

```bash
cd /d/Projects/WeaponShield-AI/web
npm test && npm run typecheck && npm run lint && npm run build
```
Expected: pass. In dev at `/` (1280 px and 360 px, both themes): the hero clip autoplays, the model loads, and person boxes appear within a few seconds; the metrics strip shows live FPS and the backend; no horizontal scroll at 360 px.

- [ ] **Step 6: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/src/features/landing web/src/content
git commit -m "feat(web): landing page with live hero demo, metrics strip and how-it-works"
```

---

### Task 10: Model card page

**Files:**
- Replace: `web/src/features/model-card/ModelCardPage.tsx`
- Create: `web/src/features/model-card/{PrCurves,ConfusionMatrix,DatasetBars,MetricsTable,ThresholdTable,FailureGallery,PendingNotice,chart-theme}.tsx`
- Test: `web/src/features/model-card/ModelCardPage.test.tsx`, `web/src/features/model-card/confusion.test.ts`

**Interfaces:**
- Consumes: `MODEL_CARD`, `ModelCard`, `ClassName`
- Produces:
  - `CLASS_COLORS: Record<ClassName, string>` = `{ person: 'var(--color-chart-1)', pistol: 'var(--color-chart-2)', rifle: 'var(--color-chart-3)', knife: 'var(--color-chart-4)' }`
  - `rowNormalise(matrix: number[][]): number[][]`, `heatStep(v: number): number` (0–6 index into the sequential blue ramp)
  - `<ModelCardPage card?={ModelCard} />` (defaults to `MODEL_CARD`; the prop exists for tests); renders `PendingNotice` + protocol when `status === 'pending'`, all result sections when `'ready'`
  - Gallery images are served from `/model-card/<file>` (Plan 5 copies them to `web/public/model-card/`)

Chart rules (dataviz method): one y-axis; 2 px lines; recessive grid; legend for ≥ 2 series plus direct end labels; crosshair tooltip on lines; per-mark tooltip on bars and cells; text in text tokens; a table next to every chart (`MetricsTable`, `ThresholdTable`, the numbers in each confusion cell).

- [ ] **Step 1: Write the failing tests**

`web/src/features/model-card/confusion.test.ts`:
```ts
import { describe, expect, it } from 'vitest';
import { heatStep, rowNormalise } from './ConfusionMatrix';

describe('confusion helpers', () => {
  it('row-normalises, leaving empty rows at zero', () => {
    expect(rowNormalise([[3, 1], [0, 0]])).toEqual([[0.75, 0.25], [0, 0]]);
  });
  it('maps 0..1 onto 7 ramp steps', () => {
    expect(heatStep(0)).toBe(0);
    expect(heatStep(0.5)).toBe(3);
    expect(heatStep(1)).toBe(6);
  });
});
```

`web/src/features/model-card/ModelCardPage.test.tsx`:
```tsx
// @vitest-environment jsdom
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { describe, expect, it } from 'vitest';
import type { ModelCard } from '@/content/model-card';
import { ModelCardPage } from './ModelCardPage';

const m = (map50: number) => ({ precision: 0.9, recall: 0.8, map50, map50_95: map50 - 0.25 });
const ready: ModelCard = {
  status: 'ready',
  model: { name: 'WeaponShield YOLO26s', file: 'ws-yolo26s.onnx', classes: ['person', 'pistol', 'rifle', 'knife'], params: '9.5M', sizeMb: 19 },
  evaluation: { protocol: ['Held-out test split'] },
  test: { per_class: { person: m(0.9), pistol: m(0.91), rifle: m(0.95), knife: m(0.78) }, weapons: m(0.88), map50_all: 0.885, map50_95_all: 0.63 },
  baseline: { name: 'v1 YOLOv8s', weapons_map50: 0.833, note: 'random split' },
  thresholds: { low: { pistol: 0.6 }, balanced: { pistol: 0.45 }, high: { pistol: 0.3 } },
  hardneg: { images: 500, fp_rate: { low: 0.01, balanced: 0.03, high: 0.07 } },
  pr_curves: { pistol: [[0, 1], [1, 0.5]] },
  confusion: { labels: ['person', 'pistol', 'rifle', 'knife', 'background'], matrix: [[90, 0, 0, 0, 5], [0, 80, 2, 0, 6], [0, 3, 70, 0, 4], [0, 0, 0, 50, 9], [4, 2, 1, 3, 0]] },
  dataset: { splits: { train: { images: 100, instances: { person: 50, pistol: 30, rifle: 20, knife: 10 } } }, sources: { 'hf-weapons': 60 } },
  gallery: [{ file: 'gallery/miss-00.jpg', kind: 'miss', class: 'knife' }],
};

const renderPage = (card: ModelCard) =>
  render(
    <MemoryRouter>
      <ModelCardPage card={card} />
    </MemoryRouter>,
  );

describe('ModelCardPage', () => {
  it('explains the pending state and the protocol', () => {
    renderPage({ ...ready, status: 'pending', test: undefined });
    expect(screen.getByText(/still training/i)).toBeInTheDocument();
    expect(screen.getByText('Held-out test split')).toBeInTheDocument();
  });

  it('renders headline metrics, the per-class table and the baseline', () => {
    renderPage(ready);
    expect(screen.getByTestId('headline-map')).toHaveTextContent('0.880');
    const table = screen.getByRole('table', { name: /per-class/i });
    expect(table).toHaveTextContent('knife');
    expect(table).toHaveTextContent('0.780');
    expect(screen.getByText(/v1 YOLOv8s/)).toBeInTheDocument();
  });

  it('shows the hard-negative false-alarm rate and the failure gallery', () => {
    renderPage(ready);
    expect(screen.getByTestId('hardneg-balanced')).toHaveTextContent('3.0%');
    expect(screen.getByRole('img', { name: /missed knife/i })).toHaveAttribute('src', '/model-card/gallery/miss-00.jpg');
  });
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /d/Projects/WeaponShield-AI/web && npx vitest run src/features/model-card`
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement the chart pieces**

`web/src/features/model-card/chart-theme.tsx`:
```tsx
import type { ClassName } from '@/content/model-card';

export const CLASS_ORDER: readonly ClassName[] = ['person', 'pistol', 'rifle', 'knife'];

export const CLASS_COLORS: Record<ClassName, string> = {
  person: 'var(--color-chart-1)',
  pistol: 'var(--color-chart-2)',
  rifle: 'var(--color-chart-3)',
  knife: 'var(--color-chart-4)',
};

export const AXIS = { stroke: 'var(--color-border)', tick: { fill: 'var(--color-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' } } as const;
export const GRID = { stroke: 'var(--ws-grid)', strokeDasharray: '0' } as const;

export function ChartTooltip({ active, payload, label, xLabel }: { active?: boolean; payload?: { name: string; value: number; color: string }[]; label?: number | string; xLabel: string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-border bg-surface-2 px-3 py-2 text-xs shadow-lg">
      <div className="mb-1 font-mono text-muted">
        {xLabel} {typeof label === 'number' ? label.toFixed(2) : label}
      </div>
      {payload.map((p) => (
        <div key={p.name} className="flex items-center gap-2">
          <span className="size-2 rounded-full" style={{ background: p.color }} />
          <span className="text-fg">{p.name}</span>
          <span className="ml-auto font-mono tabular text-fg">{p.value.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}
```

`web/src/features/model-card/PrCurves.tsx`:
```tsx
import { CartesianGrid, Label, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { ClassName, ModelCard } from '@/content/model-card';
import { AXIS, CLASS_COLORS, CLASS_ORDER, ChartTooltip, GRID } from './chart-theme';

/** Merge per-class [recall, precision] points onto a shared recall axis (0.00..1.00 step 0.01). */
function toRows(curves: NonNullable<ModelCard['pr_curves']>) {
  const rows: Record<string, number>[] = [];
  for (let i = 0; i <= 100; i++) rows.push({ recall: i / 100 });
  for (const cls of CLASS_ORDER) {
    const pts = curves[cls];
    if (!pts?.length) continue;
    for (const row of rows) {
      const r = row.recall!;
      // precision at the closest recall point (curves are sampled at 101 points by evaluate.py)
      let best = pts[0]!;
      for (const p of pts) if (Math.abs(p[0] - r) < Math.abs(best[0] - r)) best = p;
      row[cls] = best[1];
    }
  }
  return rows;
}

export function PrCurves({ curves }: { curves: NonNullable<ModelCard['pr_curves']> }) {
  const present = CLASS_ORDER.filter((c) => curves[c]?.length);
  const rows = toRows(curves);
  return (
    <figure>
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 56, bottom: 24, left: 8 }}>
            <CartesianGrid {...GRID} vertical={false} />
            <XAxis dataKey="recall" type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...AXIS}>
              <Label value="Recall" position="insideBottom" offset={-14} fill="var(--color-muted)" fontSize={11} />
            </XAxis>
            <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} width={36} {...AXIS}>
              <Label value="Precision" angle={-90} position="insideLeft" fill="var(--color-muted)" fontSize={11} />
            </YAxis>
            <Tooltip content={<ChartTooltip xLabel="Recall" />} cursor={{ stroke: 'var(--color-muted)', strokeDasharray: '3 3' }} />
            <Legend wrapperStyle={{ fontSize: 12, color: 'var(--color-muted)' }} />
            {present.map((cls: ClassName) => (
              <Line
                key={cls}
                dataKey={cls}
                name={cls}
                stroke={CLASS_COLORS[cls]}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 2, stroke: 'var(--color-surface)' }}
                isAnimationActive={false}
                label={(p: { index: number; x: number; y: number }) =>
                  p.index === Math.round(rows.length * 0.7) ? (
                    <text key={cls} x={p.x + 6} y={p.y - 6} fill="var(--color-fg)" fontSize={11} fontFamily="var(--font-mono)">
                      {cls}
                    </text>
                  ) : (
                    <g key={`${cls}-${p.index}`} />
                  )
                }
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="mt-2 text-xs text-muted">Precision–recall on the held-out test split. Higher and further right is better.</figcaption>
    </figure>
  );
}
```

`web/src/features/model-card/ConfusionMatrix.tsx`:
```tsx
const RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'] as const;

export function rowNormalise(matrix: number[][]): number[][] {
  return matrix.map((row) => {
    const sum = row.reduce((a, b) => a + b, 0);
    return row.map((v) => (sum > 0 ? v / sum : 0));
  });
}

export function heatStep(v: number): number {
  return Math.min(RAMP.length - 1, Math.max(0, Math.round(v * (RAMP.length - 1))));
}

/** Rows = true class, columns = predicted class (Ultralytics stores [predicted, true]; transpose before passing). */
export function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const norm = rowNormalise(matrix);
  return (
    <figure>
      <div className="overflow-x-auto">
        <table className="w-full border-separate border-spacing-0.5 text-xs" aria-label="Confusion matrix, rows are true classes">
          <thead>
            <tr>
              <th className="p-1 text-left font-mono font-normal text-muted">true \ predicted</th>
              {labels.map((l) => (
                <th key={l} className="p-1 font-mono font-normal text-muted">{l}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {labels.map((rowLabel, i) => (
              <tr key={rowLabel}>
                <th className="p-1 text-left font-mono font-normal text-muted">{rowLabel}</th>
                {labels.map((colLabel, j) => {
                  const v = norm[i]?.[j] ?? 0;
                  const step = heatStep(v);
                  return (
                    <td
                      key={colLabel}
                      title={`${rowLabel} → ${colLabel}: ${(v * 100).toFixed(1)}% (${matrix[i]?.[j] ?? 0})`}
                      className="h-10 min-w-12 rounded text-center font-mono tabular"
                      style={{ background: RAMP[step], color: step >= 3 ? '#ffffff' : '#0b1220' }}
                    >
                      {(v * 100).toFixed(0)}%
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <figcaption className="mt-2 text-xs text-muted">Each row sums to 100%: where the true class went. The diagonal is correct.</figcaption>
    </figure>
  );
}
```

`web/src/features/model-card/DatasetBars.tsx`:
```tsx
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { ModelCard } from '@/content/model-card';
import { AXIS, CLASS_COLORS, CLASS_ORDER, GRID } from './chart-theme';

export function DatasetBars({ dataset }: { dataset: NonNullable<ModelCard['dataset']> }) {
  const rows = Object.entries(dataset.splits).map(([split, s]) => ({ split, ...s.instances }));
  return (
    <figure>
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
            <CartesianGrid {...GRID} horizontal={false} />
            <XAxis type="number" {...AXIS} />
            <YAxis type="category" dataKey="split" width={48} {...AXIS} />
            <Tooltip cursor={{ fill: 'var(--color-surface-2)' }} contentStyle={{ background: 'var(--color-surface-2)', border: '1px solid var(--color-border)', fontSize: 12 }} />
            <Legend wrapperStyle={{ fontSize: 12, color: 'var(--color-muted)' }} />
            {CLASS_ORDER.map((c, i) => (
              <Bar key={c} dataKey={c} name={c} stackId="a" fill={CLASS_COLORS[c]} stroke="var(--color-surface)" strokeWidth={2} radius={i === CLASS_ORDER.length - 1 ? [0, 4, 4, 0] : 0} isAnimationActive={false} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="mt-2 text-xs text-muted">Labelled instances per split. People are labelled automatically by YOLO26x.</figcaption>
    </figure>
  );
}
```

- [ ] **Step 4: Implement the tables, gallery, notice and page**

`web/src/features/model-card/MetricsTable.tsx`:
```tsx
import type { ClassMetrics, ClassName } from '@/content/model-card';
import { CLASS_COLORS, CLASS_ORDER } from './chart-theme';

export function MetricsTable({ perClass, weapons }: { perClass: Partial<Record<ClassName, ClassMetrics>>; weapons: ClassMetrics }) {
  const row = (name: string, m: ClassMetrics, color?: string, strong = false) => (
    <tr key={name} className={strong ? 'font-semibold' : undefined}>
      <th scope="row" className="py-2 pr-4 text-left font-normal">
        <span className="inline-flex items-center gap-2">
          {color && <span className="size-2 rounded-full" style={{ background: color }} aria-hidden="true" />}
          {name}
        </span>
      </th>
      {[m.precision, m.recall, m.map50, m.map50_95].map((v, i) => (
        <td key={i} className="py-2 pl-4 text-right font-mono tabular">{v.toFixed(3)}</td>
      ))}
    </tr>
  );
  return (
    <table className="w-full text-sm" aria-label="Per-class results on the test split">
      <thead className="text-xs text-muted">
        <tr className="border-b border-border">
          <th className="py-2 text-left font-normal">Class</th>
          <th className="py-2 text-right font-normal">Precision</th>
          <th className="py-2 text-right font-normal">Recall</th>
          <th className="py-2 text-right font-normal">mAP50</th>
          <th className="py-2 text-right font-normal">mAP50-95</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-border">
        {CLASS_ORDER.flatMap((c) => (perClass[c] ? [row(c, perClass[c]!, CLASS_COLORS[c])] : []))}
        {row('weapons (mean)', weapons, undefined, true)}
      </tbody>
    </table>
  );
}
```

`web/src/features/model-card/ThresholdTable.tsx`:
```tsx
import type { ModelCard } from '@/content/model-card';
import { CLASS_ORDER } from './chart-theme';

const PRESETS = ['low', 'balanced', 'high'] as const;

export function ThresholdTable({ thresholds, hardneg }: { thresholds: NonNullable<ModelCard['thresholds']>; hardneg?: ModelCard['hardneg'] }) {
  return (
    <table className="w-full text-sm" aria-label="Confidence thresholds per sensitivity preset">
      <thead className="text-xs text-muted">
        <tr className="border-b border-border">
          <th className="py-2 text-left font-normal">Preset</th>
          {CLASS_ORDER.map((c) => <th key={c} className="py-2 text-right font-normal">{c}</th>)}
          {hardneg && <th className="py-2 text-right font-normal">False alarms*</th>}
        </tr>
      </thead>
      <tbody className="divide-y divide-border">
        {PRESETS.map((p) => (
          <tr key={p}>
            <th scope="row" className="py-2 text-left font-normal capitalize">{p}</th>
            {CLASS_ORDER.map((c) => (
              <td key={c} className="py-2 text-right font-mono tabular">{thresholds[p][c]?.toFixed(2) ?? '—'}</td>
            ))}
            {hardneg && (
              <td data-testid={`hardneg-${p}`} className="py-2 text-right font-mono tabular">{(hardneg.fp_rate[p] * 100).toFixed(1)}%</td>
            )}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

`web/src/features/model-card/FailureGallery.tsx`:
```tsx
import type { ModelCard } from '@/content/model-card';

export function FailureGallery({ items }: { items: NonNullable<ModelCard['gallery']> }) {
  return (
    <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
      {items.map((g) => {
        const alt = g.kind === 'miss' ? `Missed ${g.class}` : `False alarm: ${g.class}${g.conf ? ` at ${Math.round(g.conf * 100)}%` : ''}`;
        return (
          <li key={g.file} className="overflow-hidden rounded-lg border border-border bg-surface-2">
            <img src={`/model-card/${g.file}`} alt={alt} loading="lazy" className="aspect-video w-full object-cover" />
            <p className="px-2 py-1.5 font-mono text-[11px] text-muted">{alt}</p>
          </li>
        );
      })}
    </ul>
  );
}
```

`web/src/features/model-card/PendingNotice.tsx`:
```tsx
import { FlaskConical } from 'lucide-react';
import type { ModelCard } from '@/content/model-card';

export function PendingNotice({ card }: { card: ModelCard }) {
  return (
    <div className="flex gap-3 rounded-[var(--radius-card)] border border-armed/40 bg-armed/10 p-4 text-sm">
      <FlaskConical className="mt-0.5 size-5 shrink-0 text-armed" aria-hidden="true" />
      <div>
        <p className="font-medium text-fg">The WeaponShield model is still training</p>
        <p className="mt-1 text-muted">
          The app is currently running <span className="font-mono text-fg">{card.model.name}</span>. {card.model.note} Held-out results will appear here as soon as training finishes.
        </p>
      </div>
    </div>
  );
}
```

`web/src/features/model-card/ModelCardPage.tsx`:
```tsx
import type { ReactNode } from 'react';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/card';
import { Stat } from '@/components/ui/stat';
import { MODEL_CARD, type ModelCard } from '@/content/model-card';
import { ConfusionMatrix } from './ConfusionMatrix';
import { DatasetBars } from './DatasetBars';
import { FailureGallery } from './FailureGallery';
import { MetricsTable } from './MetricsTable';
import { PendingNotice } from './PendingNotice';
import { PrCurves } from './PrCurves';
import { ThresholdTable } from './ThresholdTable';

const Section = ({ title, children }: { title: string; children: ReactNode }) => (
  <Card>
    <CardHeader>
      <CardTitle>{title}</CardTitle>
    </CardHeader>
    <CardBody>{children}</CardBody>
  </Card>
);

function transpose(m: number[][]): number[][] {
  return m[0]?.map((_, j) => m.map((row) => row[j] ?? 0)) ?? [];
}

export function ModelCardPage({ card = MODEL_CARD }: { card?: ModelCard }) {
  return (
    <section className="space-y-6 py-8">
      <div>
        <p className="font-mono text-xs uppercase tracking-[0.14em] text-accent">Model card</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight sm:text-3xl">{card.model.name}</h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          What the model was trained on, how it was measured, where it fails, and what the numbers do and don’t mean.
        </p>
      </div>

      {card.status === 'pending' && <PendingNotice card={card} />}

      {card.status === 'ready' && card.test && (
        <>
          <div className="grid grid-cols-2 gap-4 rounded-[var(--radius-card)] border border-border bg-surface p-5 md:grid-cols-4">
            <Stat label="Weapons mAP50" value={card.test.weapons.map50.toFixed(3)} testId="headline-map" hint="Held-out test split" />
            <Stat label="Weapons mAP50-95" value={card.test.weapons.map50_95.toFixed(3)} />
            <Stat label="Knife mAP50" value={card.test.per_class.knife?.map50.toFixed(3) ?? '—'} hint="Hardest class" />
            <Stat label="False alarms" value={card.hardneg ? `${(card.hardneg.fp_rate.balanced * 100).toFixed(1)}%` : '—'} hint="Hard negatives, Balanced" />
          </div>
          {card.baseline && (
            <p className="text-sm text-muted">
              Baseline: {card.baseline.name} scored {card.baseline.weapons_map50.toFixed(3)} weapons mAP50 ({card.baseline.note}).
            </p>
          )}
          <div className="grid gap-6 lg:grid-cols-2">
            <Section title="Per-class results">
              <MetricsTable perClass={card.test.per_class} weapons={card.test.weapons} />
            </Section>
            {card.pr_curves && (
              <Section title="Precision–recall">
                <PrCurves curves={card.pr_curves} />
              </Section>
            )}
            {card.confusion && (
              <Section title="Confusion matrix">
                <ConfusionMatrix labels={card.confusion.labels} matrix={transpose(card.confusion.matrix)} />
              </Section>
            )}
            {card.thresholds && (
              <Section title="Sensitivity presets">
                <ThresholdTable thresholds={card.thresholds} hardneg={card.hardneg} />
                {card.hardneg && (
                  <p className="mt-3 text-xs text-muted">
                    * Share of {card.hardneg.images} images of people holding phones, tools and other look-alike objects where a weapon was reported.
                  </p>
                )}
              </Section>
            )}
            {card.dataset && (
              <Section title="Dataset">
                <DatasetBars dataset={card.dataset} />
              </Section>
            )}
          </div>
          {card.gallery && card.gallery.length > 0 && (
            <Section title="Where it fails">
              <FailureGallery items={card.gallery} />
            </Section>
          )}
        </>
      )}

      <Section title="How it is evaluated">
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-muted">
          {card.evaluation.protocol.map((p) => (
            <li key={p}>{p}</li>
          ))}
        </ul>
      </Section>

      <Section title="Intended use and limits">
        <ul className="list-disc space-y-1.5 pl-5 text-sm text-muted">
          <li>An automated second pair of eyes that supports human judgement. It is not a replacement for trained security staff.</li>
          <li>Small, distant, heavily occluded or unusual weapons are missed more often; toy and replica weapons are detected as real ones.</li>
          <li>Results are measured on still images from public datasets; accuracy on any particular camera, lens and lighting will differ.</li>
          <li>Data sources and licences are listed in the repository (training/LICENSES and the dataset’s LICENSES.md).</li>
        </ul>
      </Section>
    </section>
  );
}
```

- [ ] **Step 5: Run tests and checks; look at it**

```bash
cd /d/Projects/WeaponShield-AI/web
npx vitest run src/features/model-card && npm test && npm run typecheck && npm run lint && npm run build
```
Expected: pass. In dev, temporarily point `ModelCardPage`'s default to the `ready` fixture from the test (paste it into a scratch file and pass it via the route — do not commit) and check at 1280 px and 360 px in both themes that the PR curves have a legend + direct labels, the confusion cells are readable, and nothing overflows. Revert the scratch change.

- [ ] **Step 6: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add web/src/features/model-card
git commit -m "feat(web): model card page (pending notice, metrics, PR curves, confusion, presets, dataset, failures)"
```

---

### Task 11: Architecture page, PWA caching and removing the lab

**Files:**
- Replace: `web/src/features/architecture/ArchitecturePage.tsx`
- Create: `web/src/features/architecture/PipelineDiagram.tsx`
- Modify: `web/vite.config.ts` (PWA), `web/src/app/router.tsx` (remove `/lab`)
- Delete: `web/src/lab/`, `web/e2e/lab.spec.ts`

**Interfaces:**
- Produces: `<PipelineDiagram />` (inline SVG, `role="img"` with a text description, theme-aware via CSS variables); service worker caching `/models/*`, `/ort/*` (CacheFirst, cache `ws-runtime`) and `/demo/*` (CacheFirst with range requests, cache `ws-demo`); web app manifest with `icon.svg`

- [ ] **Step 1: Implement `web/src/features/architecture/PipelineDiagram.tsx`**

```tsx
const BOX = { rx: 10, strokeWidth: 1.5 } as const;

function Node({ x, y, w, title, sub, accent = false }: { x: number; y: number; w: number; title: string; sub: string; accent?: boolean }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={64} {...BOX} fill="var(--color-surface-2)" stroke={accent ? 'var(--color-accent)' : 'var(--color-border)'} />
      <text x={x + 14} y={y + 26} fill="var(--color-fg)" fontSize={14} fontWeight={600}>{title}</text>
      <text x={x + 14} y={y + 46} fill="var(--color-muted)" fontSize={11} fontFamily="var(--font-mono)">{sub}</text>
    </g>
  );
}

const Arrow = ({ x1, y1, x2, y2 }: { x1: number; y1: number; x2: number; y2: number }) => (
  <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--color-muted)" strokeWidth={1.5} markerEnd="url(#ws-arrow)" />
);

export function PipelineDiagram() {
  return (
    <svg viewBox="0 0 960 300" role="img" aria-labelledby="ws-diagram-title ws-diagram-desc" className="h-auto w-full min-w-[640px]">
      <title id="ws-diagram-title">WeaponShield pipeline</title>
      <desc id="ws-diagram-desc">
        Camera, video or image frames go to a Web Worker that letterboxes them and runs the YOLO26 ONNX model with WebGPU or WebAssembly. Detections go to the tracker, then the threat engine, then the overlay and alerts. The Python edge tool runs the same model on RTSP cameras and sends Telegram alerts.
      </desc>
      <defs>
        <marker id="ws-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 10 5 0 10z" fill="var(--color-muted)" />
        </marker>
      </defs>
      <text x={0} y={16} fill="var(--color-muted)" fontSize={11} fontFamily="var(--font-mono)">IN THE BROWSER</text>
      <Node x={0} y={32} w={150} title="Input" sub="camera · video · image" />
      <Node x={190} y={32} w={210} title="Web Worker" sub="letterbox → ONNX Runtime Web" accent />
      <Node x={440} y={32} w={150} title="Tracker" sub="Kalman + IoU" />
      <Node x={630} y={32} w={150} title="Threat engine" sub="K-of-N · ARMED" />
      <Node x={820} y={32} w={140} title="Overlay" sub="boxes · alerts · log" />
      <Arrow x1={150} y1={64} x2={188} y2={64} />
      <Arrow x1={400} y1={64} x2={438} y2={64} />
      <Arrow x1={590} y1={64} x2={628} y2={64} />
      <Arrow x1={780} y1={64} x2={818} y2={64} />
      <text x={200} y={118} fill="var(--color-muted)" fontSize={11} fontFamily="var(--font-mono)">WebGPU, falling back to WASM (SIMD + threads)</text>
      <line x1={0} y1={160} x2={960} y2={160} stroke="var(--color-border)" strokeDasharray="4 6" />
      <text x={0} y={190} fill="var(--color-muted)" fontSize={11} fontFamily="var(--font-mono)">NEXT TO YOUR CAMERAS (edge tool, Python)</text>
      <Node x={0} y={206} w={150} title="RTSP stream" sub="IP camera / NVR" />
      <Node x={190} y={206} w={210} title="Same ONNX model" sub="onnxruntime · CPU or CUDA" accent />
      <Node x={440} y={206} w={340} title="Same tracker + threat logic" sub="parity-tested against the browser" />
      <Node x={820} y={206} w={140} title="Telegram" sub="snapshot alert" />
      <Arrow x1={150} y1={238} x2={188} y2={238} />
      <Arrow x1={400} y1={238} x2={438} y2={238} />
      <Arrow x1={780} y1={238} x2={818} y2={238} />
    </svg>
  );
}
```

- [ ] **Step 2: Implement `web/src/features/architecture/ArchitecturePage.tsx`**

```tsx
import type { ReactNode } from 'react';
import { Card, CardBody } from '@/components/ui/card';
import { PipelineDiagram } from './PipelineDiagram';

const Block = ({ title, children }: { title: string; children: ReactNode }) => (
  <Card>
    <CardBody className="space-y-2">
      <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
      <div className="space-y-2 text-sm leading-relaxed text-muted">{children}</div>
    </CardBody>
  </Card>
);

export function ArchitecturePage() {
  return (
    <section className="space-y-6 py-8">
      <div>
        <p className="font-mono text-xs uppercase tracking-[0.14em] text-accent">How it works</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight sm:text-3xl">From a camera frame to an alert in milliseconds</h1>
      </div>
      <Card>
        <CardBody className="overflow-x-auto">
          <PipelineDiagram />
        </CardBody>
      </Card>
      <div className="grid gap-4 md:grid-cols-2">
        <Block title="Why it runs in your browser">
          <p>There is no free, always-on GPU server. Earlier versions of this project ran on Colab and on a free hosting tier, and the demo was offline more often than not.</p>
          <p>Running the model on the visitor’s own device costs nothing, can’t go down, keeps footage private, and avoids upload lag for live camera input.</p>
        </Block>
        <Block title="The model">
          <p>YOLO26, fine-tuned on public datasets for four classes: person, pistol, rifle and knife. It is exported to ONNX end-to-end, so its output is already a list of final boxes and the browser needs no extra filtering step (NMS).</p>
          <p>The same file runs in the browser (ONNX Runtime Web) and in the edge tool (ONNX Runtime).</p>
        </Block>
        <Block title="Tracking and confirmation">
          <p>A constant-velocity Kalman filter predicts where each object moves; detections are matched to tracks by overlap in two passes, high confidence first, then low confidence to keep existing tracks alive.</p>
          <p>A weapon is confirmed only once it has been seen in 4 of the last 10 frames. If its centre is inside a person’s box (widened by 20%), that person is flagged ARMED.</p>
        </Block>
        <Block title="Real CCTV cameras">
          <p>Browsers can’t open RTSP streams. The edge tool is a small Python program that reads the stream, runs the same model and logic, and sends a Telegram message with a snapshot. Tests check that it produces the same events as the browser for the same detections.</p>
        </Block>
      </div>
    </section>
  );
}
```

- [ ] **Step 3: Add PWA caching to `web/vite.config.ts`**

Add `import { VitePWA } from 'vite-plugin-pwa';` and extend `plugins`:
```ts
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      registerType: 'autoUpdate',
      injectRegister: 'auto',
      manifest: {
        name: 'WeaponShield AI',
        short_name: 'WeaponShield',
        description: 'Real-time weapon and person detection that runs entirely in your browser.',
        theme_color: '#07090c',
        background_color: '#07090c',
        display: 'standalone',
        icons: [{ src: '/icon.svg', sizes: 'any', type: 'image/svg+xml', purpose: 'any' }],
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,svg,woff2}'],
        globIgnores: ['**/*.wasm', 'ort/**', 'models/**', 'demo/**'],
        navigateFallbackDenylist: [/^\/(models|ort|demo)\//],
        runtimeCaching: [
          {
            urlPattern: ({ url }) => url.pathname.startsWith('/models/') || url.pathname.startsWith('/ort/'),
            handler: 'CacheFirst',
            options: { cacheName: 'ws-runtime', expiration: { maxEntries: 20 }, cacheableResponse: { statuses: [200] } },
          },
          {
            urlPattern: ({ url }) => url.pathname.startsWith('/demo/'),
            handler: 'CacheFirst',
            options: { cacheName: 'ws-demo', expiration: { maxEntries: 10 }, rangeRequests: true, cacheableResponse: { statuses: [200] } },
          },
        ],
      },
    }),
  ],
```

- [ ] **Step 4: Remove the lab**

Delete `web/src/lab/` and `web/e2e/lab.spec.ts`; remove the `LabPage` import and the `lab` route from `web/src/app/router.tsx`.

- [ ] **Step 5: Checks**

```bash
cd /d/Projects/WeaponShield-AI/web
npm test && npm run typecheck && npm run lint && npm run build
ls dist/sw.js dist/manifest.webmanifest
```
Expected: pass; both files exist. In `npm run preview`, load `/` twice: the second load fetches `/models/yolo26n-coco.onnx` from the service worker (DevTools → Network shows "(ServiceWorker)").

- [ ] **Step 6: Commit**

```bash
cd /d/Projects/WeaponShield-AI
git add -A web
git commit -m "feat(web): architecture page with pipeline diagram, offline model caching, remove lab page"
```

---

### Task 12: End-to-end suite, accessibility, responsiveness and Lighthouse

**Files:**
- Create: `web/e2e/landing.spec.ts`, `web/e2e/console.spec.ts`, `web/e2e/pages.spec.ts`, `web/e2e/quality.spec.ts`

**Interfaces:**
- Consumes: test ids from Tasks 6–10

- [ ] **Step 1: Write the e2e tests**

`web/e2e/landing.spec.ts`:
```ts
import { expect, test } from '@playwright/test';

test('hero demo detects people within 45 s and links to the console', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('weapon detection');
  await expect.poll(async () => Number(await page.getByTestId('hero-detections').textContent()), { timeout: 45_000 }).toBeGreaterThan(0);
  await expect(page.getByTestId('metric-backend')).toHaveText(/WebGPU|WASM/);
  await expect(page.getByTestId('metric-uploaded')).toHaveText('0');
  await page.getByRole('link', { name: /open the console/i }).click();
  await expect(page).toHaveURL(/\/console/);
});

test('is cross-origin isolated', async ({ page }) => {
  await page.goto('/');
  expect(await page.evaluate(() => crossOriginIsolated)).toBe(true);
});
```

`web/e2e/console.spec.ts`:
```ts
import { fileURLToPath } from 'node:url';
import { expect, test } from '@playwright/test';

const bus = fileURLToPath(new URL('../../fixtures/golden/images/bus.jpg', import.meta.url));

test('image mode finds people in bus.jpg', async ({ page }) => {
  await page.goto('/console?mode=image');
  await page.getByTestId('image-input').setInputFiles(bus);
  await expect.poll(async () => Number(await page.getByTestId('count-person').textContent()), { timeout: 60_000 }).toBeGreaterThanOrEqual(3);
  const download = page.waitForEvent('download');
  await page.getByTestId('image-download').click();
  expect((await download).suggestedFilename()).toBe('weaponshield-annotated.png');
});

test('live mode processes fake-camera frames', async ({ page }) => {
  await page.goto('/console?mode=live');
  await page.getByTestId('camera-toggle').click();
  await expect.poll(async () => Number(await page.getByTestId('stat-frames').textContent()), { timeout: 60_000 }).toBeGreaterThan(5);
  await expect(page.getByTestId('backend')).toHaveText(/WebGPU|WASM/);
  await page.getByTestId('camera-toggle').click();
});

test('video mode plays a demo clip with detection', async ({ page }) => {
  await page.goto('/console?mode=video');
  await page.getByTestId('video-play').click();
  await expect.poll(async () => Number(await page.getByTestId('stat-frames').textContent()), { timeout: 60_000 }).toBeGreaterThan(5);
  await expect.poll(async () => Number(await page.getByTestId('stat-persons').textContent()), { timeout: 30_000 }).toBeGreaterThan(0);
});
```

`web/e2e/pages.spec.ts`:
```ts
import { expect, test } from '@playwright/test';

test('model card explains its state', async ({ page }) => {
  await page.goto('/model');
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  await expect(page.getByText(/how it is evaluated/i)).toBeVisible();
});

test('architecture page shows the diagram', async ({ page }) => {
  await page.goto('/architecture');
  await expect(page.getByRole('img', { name: /weaponshield pipeline/i })).toBeVisible();
});

test('unknown routes show the 404 page', async ({ page }) => {
  await page.goto('/definitely-not-here');
  await expect(page.getByText(/doesn’t exist/)).toBeVisible();
});
```

`web/e2e/quality.spec.ts`:
```ts
import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

const ROUTES = ['/', '/console', '/model', '/architecture'];

for (const route of ROUTES) {
  test(`no serious accessibility violations on ${route}`, async ({ page }) => {
    await page.goto(route);
    await page.waitForLoadState('networkidle');
    // Demo videos are muted and have no audio track, so captions (video-caption) do not apply.
    const results = await new AxeBuilder({ page }).disableRules(['video-caption']).analyze();
    const bad = results.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical');
    expect(bad.map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
  });

  test(`no horizontal scroll at 360px on ${route}`, async ({ page }) => {
    await page.setViewportSize({ width: 360, height: 740 });
    await page.goto(route);
    await page.waitForLoadState('networkidle');
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(0);
  });

  test(`light theme renders ${route}`, async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem('ws-theme', 'light'));
    await page.goto(route);
    expect(await page.evaluate(() => document.documentElement.dataset.theme)).toBe('light');
  });
}
```

- [ ] **Step 2: Run the suite and fix real failures**

```bash
cd /d/Projects/WeaponShield-AI/web
npm run build && npm run e2e
```
Expected: all pass. Fix causes in the app, never by loosening assertions: e.g. a colour-contrast violation means adjusting the token or text size; horizontal overflow means finding the overflowing element (`document.querySelectorAll('*')` with `scrollWidth > clientWidth`) and constraining it. The architecture diagram's horizontal scroll lives inside its own `overflow-x-auto` card and does not count as page overflow.

- [ ] **Step 3: Lighthouse check**

```bash
cd /d/Projects/WeaponShield-AI/web
npm run preview -- --port 4173 &
npx --yes lighthouse@latest http://localhost:4173/ --quiet --chrome-flags="--headless=new" --only-categories=performance,accessibility,best-practices --output=json --output-path=../.lighthouse-home.json
node -e "const r=require('../.lighthouse-home.json');for(const [k,v] of Object.entries(r.categories))console.log(k,Math.round(v.score*100))"
kill %1
```
Expected: each score ≥ 90. If performance is below 90, the usual causes are the hero video (`preload="auto"` on a large clip — re-encode smaller or use `preload="metadata"` until the model is ready) and the model download competing with first paint (it starts after the hero renders; keep it that way). Record the three scores in the commit message. Delete `.lighthouse-home.json` afterwards (do not commit it).

- [ ] **Step 4: Commit, check attribution, push**

```bash
cd /d/Projects/WeaponShield-AI
git add web/e2e
git commit -m "test(web): e2e for landing, console modes, pages, accessibility, responsiveness and themes"
git log origin/v2..HEAD --format=%B | grep -i -E "co-authored|claude|generated with" || echo "attribution check: clean"
git push origin v2
```
Expected: `attribution check: clean`; the `web` CI workflow passes (`gh run watch` as in Plan 1 Task 12).

---

## Notes for later plans

- Plan 5 replaces `web/src/content/model-card.json` with `status: "ready"` data built from `metrics.json` + `stats.json` (confusion matrix stays in Ultralytics' `[predicted, true]` layout; the page transposes it), copies `gallery/*.jpg` into `web/public/model-card/gallery/`, swaps `ACTIVE_MODEL`, and retunes `PRESETS`.
- Plan 5 also slims the duplicate ORT wasm emitted into `dist/assets` and versions model filenames (the `/models/` cache header is `immutable`).
