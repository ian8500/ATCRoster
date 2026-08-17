import { expect, test } from "@playwright/test";
import crypto from "node:crypto";

const password = process.env.ATCROSTER_E2E_PASSWORD || "Test-ATCRoster-2026!";
const mfaSecret = process.env.ATCROSTER_E2E_MFA_SECRET || "JBSWY3DPEHPK3PXP";

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

async function signIn(page, username = "lba.admin") {
  await page.goto("/login");
  await page.getByLabel(/username/i).fill(username);
  await page.getByRole("textbox", { name: "Password" }).fill(password);
  await Promise.all([
    page.waitForURL("**/login/mfa"),
    page.getByRole("button", { name: /sign in|login/i }).click(),
  ]);
  const verificationCode = page.getByLabel(/code/i);
  const verify = page.getByRole("button", { name: /verify|continue/i });
  await verificationCode.fill(totp(mfaSecret));
  await verify.click();
  await page.waitForTimeout(500);
  if (page.url().endsWith("/login/mfa")) {
    // The acceptance app rejects replayed TOTP values. This can occur when an
    // earlier serial browser scenario used the same deterministic test user.
    const nextCodeDelay = 30_100 - (Date.now() % 30_000);
    await page.waitForTimeout(nextCodeDelay);
    await verificationCode.fill(totp(mfaSecret));
    await verify.click();
  }
  await expect(page).not.toHaveURL(/\/login\/mfa$/);
}

test("shared application UI remains polished, legible, and usable across responsive surfaces", async ({ page }) => {
  // MFA replay protection can require waiting almost one full 30-second TOTP
  // interval when an earlier serial scenario used this acceptance identity.
  // Keep the assertions strict while giving that deliberate recovery path
  // enough time to complete before the test-level timeout expires.
  test.setTimeout(60_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  await signIn(page);
  await test.step("module launcher cards", async () => {
    await page.goto("/modules");
    const launcher = page.locator(".module-launcher");
    const cards = page.locator(".module-launcher__grid > .module-card");
    await expect(launcher).toBeVisible();
    await expect(page.getByRole("link", { name: "Roster", exact: true })).toBeVisible();
    expect(await cards.count()).toBeGreaterThanOrEqual(2);

    const presentation = await cards.evaluateAll((nodes) => nodes.map((node) => {
      const style = getComputedStyle(node);
      const rect = node.getBoundingClientRect();
      return {
        label: node.textContent?.trim(),
        background: style.backgroundColor,
        radius: Number.parseFloat(style.borderTopLeftRadius),
        height: rect.height,
        contained: rect.left >= -1 && rect.right <= window.innerWidth + 1,
      };
    }));
    expect(presentation.every((card) => card.label && card.background !== "transparent" && card.background !== "rgba(0, 0, 0, 0)")).toBe(true);
    expect(presentation.every((card) => card.radius > 0 && card.height >= 80 && card.contained)).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
  });

  await test.step("shared form and table surfaces", async () => {
    await page.goto("/requests");
    await expect(page.getByRole("heading", { name: "Shift requests" })).toBeVisible();
    const dateInput = page.getByLabel("Date");
    const table = page.locator(".table-responsive > table").first();
    await expect(dateInput).toBeVisible();
    await expect(table).toBeVisible();
    const requestSurface = await dateInput.evaluate((input) => {
      const card = input.closest(".card");
      const table = document.querySelector(".table-responsive > table");
      const tableFrame = table?.parentElement;
      const inputStyle = getComputedStyle(input);
      const cardStyle = card ? getComputedStyle(card) : null;
      const tableFrameStyle = tableFrame ? getComputedStyle(tableFrame) : null;
      return {
        inputBackground: inputStyle.backgroundColor,
        inputRadius: Number.parseFloat(inputStyle.borderTopLeftRadius),
        inputHeight: input.getBoundingClientRect().height,
        cardBackground: cardStyle?.backgroundColor,
        tableOverflow: tableFrameStyle?.overflowX,
      };
    });
    expect(requestSurface.inputBackground).not.toBe("transparent");
    expect(requestSurface.inputBackground).not.toBe("rgba(0, 0, 0, 0)");
    expect(requestSurface.inputRadius).toBeGreaterThan(0);
    expect(requestSurface.inputHeight).toBeGreaterThanOrEqual(32);
    expect(requestSurface.cardBackground).not.toBe("transparent");
    expect(requestSurface.cardBackground).not.toBe("rgba(0, 0, 0, 0)");
    expect(["auto", "scroll"]).toContain(requestSurface.tableOverflow);
  });

  await test.step("empty state", async () => {
    await page.goto("/reports/operational-activity?scope=individual");
    const emptyState = page.locator(".empty-state").filter({
      hasText: "Select a controller to view their operational activity figures.",
    });
    await expect(emptyState).toBeVisible();
    const emptyPresentation = await emptyState.evaluate((node) => {
      const style = getComputedStyle(node);
      return {
        display: style.display,
        padding: Number.parseFloat(style.paddingTop),
        textAlign: style.textAlign,
        background: style.backgroundColor,
      };
    });
    expect(emptyPresentation.display).toBe("grid");
    expect(emptyPresentation.padding).toBeGreaterThan(0);
    expect(emptyPresentation.textAlign).toBe("center");
    expect(emptyPresentation.background).not.toBe("transparent");
    expect(emptyPresentation.background).not.toBe("rgba(0, 0, 0, 0)");
  });

  await test.step("mobile module launcher and navigation", async () => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/reports/operational-activity?scope=individual");
    await expect(page.getByRole("heading", { name: "Operational activity" })).toBeVisible();
    await expect(page.locator(".operational-report-filters")).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);

    await page.goto("/modules");
    const cards = page.locator(".module-launcher__grid > .module-card");
    await expect(cards.first()).toBeVisible();
    const mobileLayout = await cards.evaluateAll((nodes) => ({
      cardsWithinViewport: nodes.every((node) => {
        const rect = node.getBoundingClientRect();
        return rect.left >= -1 && rect.right <= window.innerWidth + 1;
      }),
      pageFitsViewport: document.documentElement.scrollWidth <= window.innerWidth + 1,
    }));
    expect(mobileLayout.cardsWithinViewport).toBe(true);
    expect(mobileLayout.pageFitsViewport).toBe(true);

    const menu = page.getByRole("button", { name: "Menu" });
    const navigation = page.locator("#primary-navigation");
    await expect(menu).toHaveAttribute("aria-expanded", "false");
    await menu.click();
    await expect(menu).toHaveAttribute("aria-expanded", "true");
    await expect(navigation).toHaveClass(/is-open/);
    await expect(navigation.getByRole("link", { name: "Home", exact: true })).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(menu).toHaveAttribute("aria-expanded", "false");
    await expect(navigation).not.toHaveClass(/is-open/);
    await expect(menu).toBeFocused();
  });
});
