# Accessibility (psd-to-svg frontend)

Target: **WCAG 2.2 Level AA**.

## Automated testing

Two layers:

- **Static** — `eslint-plugin-jsx-a11y` runs as part of `npm run lint`.
- **Runtime** — [Playwright](https://playwright.dev/) +
  [`@axe-core/playwright`](https://github.com/dequelabs/axe-core-npm):

```bash
npm run test:e2e            # boots the Vite dev server and runs the axe scans
npx playwright test --ui    # interactive runner
```

Specs live in [`e2e/`](./e2e). They scan the WCAG A/AA tag set
(`wcag2a, wcag2aa, wcag21a, wcag21aa, wcag22aa`) over the whole single-page app and over the
playground with the (default-collapsed) **Color Mapping** card expanded.

> First run downloads Chromium: `npx playwright install chromium`.
> The embedded YouTube player is third-party content; axe descends into the cross-origin
> iframe and reports YouTube's own issues, so the scan excludes `iframe`.

## Already in good shape

- Collapsible playground cards use `aria-expanded` + `aria-controls` and mark collapsed panels
  `inert`. The SVG graphics (`CarGraphic`) inject `role="img"`/`role="group"` + `aria-label`
  and hide the decorative base image. The dash-array options are keyboard-operable
  (`role="button"` + Enter/Space handlers).

## Manual test plan (the part axe can't automate)

1. **Keyboard only:** Tab through the nav anchors, Upload/Download buttons, the playground
   collapse toggles, dash-array options (Enter/Space), and the sliders (arrow keys) — all with
   a visible focus ring. Tab into the code panels and scroll with arrow keys.
2. **Screen reader** (NVDA / VoiceOver): headings form a logical outline; sliders announce
   their label and value; the car graphics announce their labels; collapsed cards announce
   expanded/collapsed state.
3. **Conversion flow:** upload a `.psd`, confirm the processing spinner and resulting preview
   `<img alt="Processed SVG">` are announced, and Download becomes enabled.
4. **Zoom / reflow:** 200%/400% zoom — the grid layouts reflow without loss of content.
5. **Contrast:** spot-check **dark mode** (`prefers-color-scheme: dark`) too — axe only scanned
   the default light theme.
6. **Motion:** the D3 animation and card transitions — consider `prefers-reduced-motion`.
