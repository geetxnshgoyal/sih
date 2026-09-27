import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";

test("complete customer/staff charge conversation, export, and confirmed reset", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page
    .getByLabel("Service type", { exact: true })
    .selectOption("banking");
  await expect(page.getByRole("log")).not.toContainText("unexpected charge");
  for (let i = 0; i < 5; i++) {
    await page
      .getByRole("button", {
        name: i === 0 ? "Start guided demo" : "Select next phrase",
      })
      .click();
    await expect(page.getByRole("log").locator("article")).toHaveCount(i);
    await page
      .getByRole("button", {
        name: i % 2 === 0 ? "Send to staff" : "Send to user",
        exact: true,
      })
      .click();
    await expect(page.getByRole("log").locator("article")).toHaveCount(i + 1);
  }
  await expect(page.getByLabel("Guided demo")).toContainText(
    "Conversation complete",
  );
  await expect(page.getByRole("log")).toContainText(
    "Please show me the date and amount",
  );
  await expect(page.getByRole("log")).toContainText("₹250");
  await expect(page.getByRole("log")).toContainText(
    "I need to review the charge",
  );
  const pending = page.waitForEvent("download");
  await page.getByLabel("Download conversation").click();
  const download = await pending;
  const text = await readFile((await download.path())!, "utf8");
  expect(text).toContain("SERVICE USER");
  expect(text).toContain("STAFF");
  expect(text).toContain("₹250");
  await page.getByRole("button", { name: "New conversation" }).click();
  await page.getByRole("button", { name: "Keep conversation" }).click();
  await expect(page.getByRole("log").locator("article")).toHaveCount(5);
  await page.getByRole("button", { name: "New conversation" }).click();
  await page.getByRole("button", { name: "Clear and start" }).click();
  await expect(page.getByRole("log").locator("article")).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("Hindi phrase selection, library search, typed staff response and source labels", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByLabel("Service type", { exact: true })
    .selectOption("banking");
  await page.getByLabel("Message language").selectOption("hi-IN");
  await page.getByRole("button", { name: "Start guided demo" }).click();
  await expect(page.getByLabel("Review or type a message")).toHaveValue(
    /मेरे खाते/,
  );
  await page
    .getByRole("button", { name: "Send to staff", exact: true })
    .click();
  await expect(page.getByRole("log")).toContainText("मेरे खाते");
  await page
    .getByRole("button", { name: "Service staff", exact: true })
    .click();
  await page.getByLabel("Review or type a message").fill("कृपया तारीख दिखाएँ।");
  await page.getByRole("button", { name: "Send to user", exact: true }).click();
  await expect(page.getByRole("log")).toContainText("Typed text");
  await page
    .getByRole("button", { name: "Phrase library", exact: true })
    .click();
  await page.getByLabel("Find a phrase").fill("complaint");
  await expect(page.locator(".library-phrase")).toHaveCount(1);
  await page.locator(".library-phrase").click();
  await expect(page.getByLabel("Review or type a message")).toHaveValue(
    /शिकायत/,
  );
});

test("unsupported microphone and missing voice never fabricate messages", async ({
  page,
}) => {
  await page.addInitScript(() => {
    Object.defineProperty(window, "SpeechRecognition", { value: undefined });
    Object.defineProperty(window, "webkitSpeechRecognition", {
      value: undefined,
    });
    Object.defineProperty(window.speechSynthesis, "getVoices", {
      value: () => [],
    });
  });
  await page.goto("/");
  await page
    .getByLabel("Service type", { exact: true })
    .selectOption("banking");
  await page.getByRole("button", { name: "Use microphone" }).click();
  await expect(page.getByRole("status")).toContainText("unavailable");
  await expect(page.getByRole("log").locator("article")).toHaveCount(0);
  await expect(page.getByLabel("Review or type a message")).toHaveValue("");
  await page.getByLabel("Review or type a message").fill("Please help.");
  await page
    .getByRole("button", { name: "Send to staff", exact: true })
    .click();
  await page.getByRole("button", { name: "Read aloud", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("No matching voice");
});

test("recorded sign playback is real, and unmatched words do not substitute a greeting", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByLabel("Service type", { exact: true })
    .selectOption("banking");
  await page.getByLabel("Review or type a message").fill("Hello");
  await page
    .getByRole("button", { name: "Send to staff", exact: true })
    .click();
  await page.getByRole("button", { name: "Explore recorded signs" }).click();
  await expect(page.locator(".playback-panel canvas")).toBeVisible();
  await expect(page.locator(".playback-panel")).toContainText(
    "does not translate full ISL grammar",
  );
  await page.getByLabel("Review or type a message").fill("UnrecognisableXYZ");
  await page
    .getByRole("button", { name: "Send to staff", exact: true })
    .click();
  await expect(page.locator(".playback-panel")).toContainText(
    "No sign recordings match",
  );
  await expect(page.locator(".playback-panel canvas")).toHaveCount(0);
});

test.describe("model failure without cache", () => {
  test.use({ serviceWorkers: "block" });
  test("missing model reports an error and preserves the phrase fallback", async ({
    page,
  }) => {
    await page.route("**/model/model.json", (route) => route.abort());
    await page.goto("/");
    await page
      .getByLabel("Service type", { exact: true })
      .selectOption("banking");
    await page
      .getByRole("button", { name: "Try sign recognition Experimental" })
      .click();
    await expect(page.locator(".sign-panel")).toContainText("Model failed", {
      timeout: 20000,
    });
    await expect(page.getByRole("log").locator("article")).toHaveCount(0);
    await page.getByRole("button", { name: "Start guided demo" }).click();
    await page
      .getByRole("button", { name: "Send to staff", exact: true })
      .click();
    await expect(page.getByRole("log")).toContainText("unexpected charge");
  });
});

test("cached production app reloads offline and completes both directions", async ({
  page,
  context,
}) => {
  await page.goto("/");
  await page
    .getByLabel("Service type", { exact: true })
    .selectOption("banking");
  await page.evaluate(() => navigator.serviceWorker.ready);
  await page.reload();
  await page.waitForFunction(() => !!navigator.serviceWorker.controller);
  // Wait for the actual shell dependencies, not an arbitrary time delay.
  await page.waitForFunction(async () => {
    const urls = [
      ...document.querySelectorAll('script[src],link[rel="stylesheet"]'),
    ].map((el) => el.getAttribute("src") || el.getAttribute("href")!);
    return (
      await Promise.all(
        urls.map((url) => caches.match(new URL(url, location.href))),
      )
    ).every(Boolean);
  });
  await context.setOffline(true);
  await page.reload();
  await expect(
    page.getByText("You’re offline.", { exact: false }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Start guided demo" }).click();
  await page
    .getByRole("button", { name: "Send to staff", exact: true })
    .click();
  await page.getByRole("button", { name: "Select next phrase" }).click();
  await page.getByRole("button", { name: "Send to user", exact: true }).click();
  await expect(page.getByRole("log").locator("article")).toHaveCount(2);
});

test("mobile, enlarged text, keyboard access and navigation stay usable", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page
    .getByLabel("Service type", { exact: true })
    .selectOption("banking");
  await page.keyboard.press("Tab");
  await expect(
    page.getByRole("link", { name: "Skip to conversation" }),
  ).toBeFocused();
  await page.keyboard.press("Enter");
  await page.getByRole("button", { name: "Larger text" }).click();
  await page.getByRole("button", { name: "Start guided demo" }).click();
  await page
    .getByRole("button", { name: "Send to staff", exact: true })
    .click();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.getByRole("button", { name: "Access & capabilities" }).click();
  await expect(
    page.getByRole("heading", { name: "Access, without assumptions." }),
  ).toBeVisible();
  await page.setViewportSize({ width: 320, height: 720 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});

test("counter and library pass automated WCAG A/AA accessibility checks", async ({
  page,
}) => {
  const { default: AxeBuilder } = await import("@axe-core/playwright");
  await page.goto("/");
  await page
    .getByLabel("Service type", { exact: true })
    .selectOption("banking");
  for (const name of [
    "Service counter",
    "Phrase library",
    "Access & capabilities",
  ]) {
    await page.getByRole("button", { name, exact: true }).click();
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(
      results.violations.map((issue) => ({
        id: issue.id,
        nodes: issue.nodes.map((node) => node.target),
      })),
    ).toEqual([]);
  }
});

test("camera refusal leaves the conversation empty and the phrase board usable", async ({
  page,
}) => {
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = async () => {
      throw new DOMException("Camera permission denied", "NotAllowedError");
    };
  });
  await page.goto("/");
  await page
    .getByLabel("Service type", { exact: true })
    .selectOption("banking");
  await page
    .getByRole("button", { name: "Try sign recognition Experimental" })
    .click();
  await page
    .getByRole("button", { name: "Start camera", exact: true })
    .click({ timeout: 20000 });
  await expect(page.locator(".sign-panel")).toContainText(
    "Camera permission denied",
  );
  await expect(page.getByRole("log").locator("article")).toHaveCount(0);
  await page.getByRole("button", { name: "Start guided demo" }).click();
  await page
    .getByRole("button", { name: "Send to staff", exact: true })
    .click();
  await expect(page.getByRole("log")).toContainText("unexpected charge");
});

test("other speech languages use an explicit English phrase fallback", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByLabel("Message language").selectOption("ta-IN");
  await expect(
    page.getByText("Written phrases are available in English and Hindi.", {
      exact: false,
    }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Start guided demo" }).click();
  await page
    .getByRole("button", { name: "Send to staff", exact: true })
    .click();
  await expect(page.locator(".message-display > p")).toHaveAttribute(
    "lang",
    "en-IN",
  );
  await page.getByLabel("Review or type a message").fill("நன்றி");
  await page
    .getByRole("button", { name: "Send to staff", exact: true })
    .click();
  await expect(page.locator(".message-display > p")).toHaveAttribute(
    "lang",
    "ta-IN",
  );
});

test("opens as a general service app and completes each service conversation", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByLabel("Service type", { exact: true })).toHaveValue(
    "general",
  );
  await expect(page.getByText("Tertiary services workspace")).toBeVisible();
  for (const service of [
    "general",
    "banking",
    "hospitality",
    "retail",
    "transport",
    "public",
  ]) {
    await page
      .getByLabel("Service type", { exact: true })
      .selectOption(service);
    for (let step = 0; step < 5; step++) {
      await page
        .getByRole("button", {
          name: step === 0 ? "Start guided demo" : "Select next phrase",
          exact: true,
        })
        .click();
      await page
        .getByRole("button", {
          name: step % 2 === 0 ? "Send to staff" : "Send to user",
          exact: true,
        })
        .click();
    }
    await expect(page.getByLabel("Guided demo")).toContainText(
      "Conversation complete",
    );
  }
  await expect(page.getByRole("log").locator("article")).toHaveCount(30);
  await expect(page.getByRole("log")).toContainText("reservation");
  await expect(page.getByRole("log")).toContainText("return an item");
  await expect(page.getByRole("log")).toContainText("destination");
  await expect(page.getByRole("log")).toContainText("documents");
});

test("service change keeps drafts, filters phrases, and preserves their original context", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByLabel("Service type", { exact: true }).selectOption("retail");
  await page.getByRole("button", { name: "Start guided demo" }).click();
  await page
    .getByLabel("Service type", { exact: true })
    .selectOption("hospitality");
  await expect(page.getByLabel("Review or type a message")).toHaveValue(
    /return an item/,
  );
  await page
    .getByRole("button", { name: "Send to staff", exact: true })
    .click();
  await expect(page.getByRole("log")).toContainText("Retail");
  await page
    .getByRole("button", { name: "Phrase library", exact: true })
    .click();
  await expect(page.locator(".library")).toContainText("reservation");
  await expect(page.locator(".library")).not.toContainText("unexpected charge");
  await expect(page.locator(".library")).not.toContainText("return an item");
});
