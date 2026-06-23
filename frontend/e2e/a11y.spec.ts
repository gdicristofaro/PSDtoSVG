import { test, expect, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

// WCAG 2.2 Level AA (plus the earlier A/AA baselines axe maps onto these tags).
const WCAG_TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa'];

async function expectNoViolations(page: Page, context?: string) {
  const results = await new AxeBuilder({ page })
    .withTags(WCAG_TAGS)
    // The embedded YouTube player is third-party content we don't control; axe descends into
    // the cross-origin iframe and reports YouTube's own violations, so exclude it.
    .exclude('iframe')
    .analyze();
  expect(results.violations, formatViolations(results.violations, context)).toEqual([]);
}

function formatViolations(
  violations: Awaited<ReturnType<AxeBuilder['analyze']>>['violations'],
  context?: string
) {
  if (violations.length === 0) return 'no violations';
  const header = context ? `axe violations (${context}):` : 'axe violations:';
  return [
    header,
    ...violations.map((v) => `  - [${v.impact}] ${v.id}: ${v.help} (${v.nodes.length} node(s))`),
  ].join('\n');
}

test.describe('psd-to-svg accessibility', () => {
  // The whole app is a single page; every section is rendered at once.
  test('full page (all sections) has no axe violations', async ({ page }) => {
    await page.goto('./');
    await expect(page.getByRole('heading', { name: 'Convert' })).toBeVisible();
    await expectNoViolations(page, 'full page');
  });

  // The "Color Mapping" playground card is collapsed (inert) by default; expand it so its
  // controls are part of the accessibility tree, then scan.
  test('playground with all controls expanded has no axe violations', async ({ page }) => {
    await page.goto('./');
    await page.getByRole('button', { name: 'Color Mapping' }).click();
    await expect(page.getByRole('button', { name: 'Color Mapping' })).toHaveAttribute(
      'aria-expanded',
      'true'
    );
    await expectNoViolations(page, 'playground expanded');
  });
});
