import { expect, test } from "@playwright/test";
import crypto from "node:crypto";

const password = process.env.ATCROSTER_E2E_PASSWORD || "Test-ATCRoster-2026!";
const mfaSecret = process.env.ATCROSTER_E2E_MFA_SECRET || "JBSWY3DPEHPK3PXP";
const rosterMonth = process.env.ATCROSTER_E2E_ROSTER_MONTH
  || new Date().toISOString().slice(0, 7);

function totp(secret) {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
  let bits = "";
  for (const character of secret.replace(/=+$/, "")) bits += alphabet.indexOf(character).toString(2).padStart(5, "0");
  const key = Buffer.from(bits.match(/.{1,8}/g).map(value => parseInt(value.padEnd(8, "0"), 2)));
  const counter = Buffer.alloc(8); counter.writeBigUInt64BE(BigInt(Math.floor(Date.now() / 30_000)));
  const hash = crypto.createHmac("sha1", key).update(counter).digest();
  const offset = hash[19] & 15;
  return String(((hash.readUInt32BE(offset) & 0x7fffffff) % 1_000_000)).padStart(6, "0");
}

let staffCookies;

test.describe.configure({ mode: "serial" });

test("staff roster view is read-only without misleading editor controls", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel(/username/i).fill("lba.atco01");
  await page.getByRole("textbox", { name: "Password" }).fill(password);
  await Promise.all([
    page.waitForURL("**/login/mfa"),
    page.getByRole("button", { name: /sign in|login/i }).click(),
  ]);
  await page.getByLabel(/code/i).fill(totp(mfaSecret));
  await Promise.all([
    page.waitForURL((url) => !url.pathname.endsWith("/login/mfa")),
    page.getByRole("button", { name: /verify|continue/i }).click(),
  ]);
  await page.goto(`/roster/${rosterMonth}`);
  staffCookies = await page.context().cookies();

  await expect(page.getByRole("link", { name: "Skip to main content" })).toBeVisible();
  await expect(page.getByRole("main")).toBeVisible();
  await expect(page.getByRole("group", { name: "Roster zoom" })).toBeVisible();
  const roster = page.locator("table.roster");
  expect(await roster.locator("thead th[scope='col']").count()).toBeGreaterThan(7);
  expect(await roster.locator("tbody th[scope='row']").count()).toBeGreaterThan(0);
  await expect(page.locator("[data-roster-readiness]")).toHaveCount(0);
  await expect(page.locator("[data-roster-command-open]")).toHaveCount(0);
  await expect(page.locator("[data-roster-command-palette]")).toHaveCount(0);
  await expect(page.locator("[data-roster-inspector]")).toHaveCount(0);
  await expect(page.locator("[data-roster-undo]")).toHaveCount(0);
  await expect(page.locator(".cell.editable")).toHaveCount(0);
});

test("roster command bar stays compact without clipping its controls", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.context().addCookies(staffCookies);
  await page.goto(`/roster/${rosterMonth}`);

  const heading = page.locator(".roster-workspace-heading > h2");
  const toolbar = page.locator(".roster-month-nav");
  const overview = page.locator(".roster-overview");
  const zoom = page.getByRole("group", { name: "Roster zoom" });
  await expect(toolbar).toBeVisible();
  await expect(toolbar.getByRole("link", { name: "Export CSV" })).toBeVisible();
  await expect(toolbar.getByRole("button", { name: "Print" })).toBeVisible();
  await expect(zoom).toBeVisible();
  await expect(zoom.getByRole("button")).toHaveCount(4);

  const compactLayout = await toolbar.evaluate((node) => {
    const frame = node.getBoundingClientRect();
    const children = [...node.children].filter((child) => {
      const rect = child.getBoundingClientRect();
      return getComputedStyle(child).display !== "none" && rect.width > 0 && rect.height > 0;
    });
    const rows = new Set(children.map((child) => Math.round(child.getBoundingClientRect().top)));
    const tallestChild = Math.max(...children.map((child) => child.getBoundingClientRect().height));
    return {
      clientWidth: node.clientWidth,
      scrollWidth: node.scrollWidth,
      rowCount: rows.size,
      height: frame.height,
      tallestChild,
    };
  });
  expect(compactLayout.scrollWidth).toBeLessThanOrEqual(compactLayout.clientWidth + 1);
  expect(compactLayout.rowCount).toBe(1);
  expect(compactLayout.height).toBeLessThanOrEqual(compactLayout.tallestChild + 32);

  const [headingBox, toolbarBox, overviewBox] = await Promise.all([
    heading.boundingBox(), toolbar.boundingBox(), overview.boundingBox(),
  ]);
  expect(headingBox).not.toBeNull();
  expect(toolbarBox).not.toBeNull();
  expect(overviewBox).not.toBeNull();
  expect(headingBox.x + headingBox.width).toBeLessThanOrEqual(overviewBox.x + 1);
  expect(toolbarBox.y).toBeGreaterThanOrEqual(
    Math.max(headingBox.y + headingBox.height, overviewBox.y + overviewBox.height) - 1,
  );
});

test("mobile navigation opens, closes, and returns focus predictably", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.context().addCookies(staffCookies);
  await page.goto(`/roster/${rosterMonth}`);

  const toolbar = page.locator(".roster-month-nav");
  await expect(toolbar).toBeVisible();
  const mobileToolbar = await toolbar.evaluate((node) => {
    const controls = [...node.querySelectorAll("a, button")]
      .filter((control) => {
        const rect = control.getBoundingClientRect();
        return getComputedStyle(control).display !== "none" && rect.width > 0 && rect.height > 0;
      });
    return {
      clientWidth: node.clientWidth,
      scrollWidth: node.scrollWidth,
      controlsWithinViewport: controls.every((control) => {
        const rect = control.getBoundingClientRect();
        return rect.left >= -1 && rect.right <= window.innerWidth + 1;
      }),
    };
  });
  expect(mobileToolbar.scrollWidth).toBeLessThanOrEqual(mobileToolbar.clientWidth + 1);
  expect(mobileToolbar.controlsWithinViewport).toBe(true);

  const menu = page.getByRole("button", { name: "Menu" });
  const navigation = page.locator("#primary-navigation");
  await expect(menu).toBeVisible();
  await expect(menu).toHaveAttribute("aria-expanded", "false");
  await menu.click();
  await expect(menu).toHaveAttribute("aria-expanded", "true");
  await expect(navigation).toHaveClass(/is-open/);
  await page.keyboard.press("Escape");
  await expect(menu).toHaveAttribute("aria-expanded", "false");
  await expect(navigation).not.toHaveClass(/is-open/);
  await expect(menu).toBeFocused();
});

test("roster keyboard navigation moves between editable assignments", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel(/username/i).fill("lba.editor");
  await page.getByRole("textbox", { name: "Password" }).fill(password);
  await Promise.all([
    page.waitForURL("**/login/mfa"),
    page.getByRole("button", { name: /sign in|login/i }).click(),
  ]);
  await page.getByLabel(/code/i).fill(totp(mfaSecret));
  await Promise.all([
    page.waitForURL((url) => !url.pathname.endsWith("/login/mfa")),
    page.getByRole("button", { name: /verify|continue/i }).click(),
  ]);
  await page.goto(`/roster/${rosterMonth}`);

  const selected = page.locator(".cell.editable")
    .filter({ has: page.locator("[data-roster-shift-select]") }).first();
  const originalDay = await selected.getAttribute("data-roster-day");
  await selected.evaluate((cell) => cell.click());
  await expect(selected).toHaveClass(/is-selected/);
  await expect(page.locator("[data-roster-inspector]")).toBeVisible();
  await expect(page.locator("[data-roster-inspector]")).toHaveAttribute("aria-live", "polite");
  await page.keyboard.press("ArrowRight");
  const moved = page.locator(".cell.is-selected");
  await expect(moved).toHaveCount(1);
  await expect(moved).not.toHaveAttribute("data-roster-day", originalDay);
  await expect(moved.locator("[data-roster-shift-select]")).toBeFocused();
});
