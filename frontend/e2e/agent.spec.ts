import { expect, type Page, test } from '@playwright/test'

async function ask(page: Page, question: string) {
  await expect(page.getByTestId('connection-status')).toHaveText('Connected')
  const before = await page.getByTestId('assistant-message').count()
  await page.getByTestId('question-input').fill(question)
  await page.getByTestId('send-button').click()
  const msg = page.getByTestId('assistant-message').nth(before)
  // Both endpoints end with response.end + a client-side checksum verification.
  await expect(msg.getByTestId('integrity-badge')).toHaveText('stream verified')
  return msg
}

/** Print the inspector's reasoning timeline as rendered in the UI (the run log). */
async function logTimeline(page: Page, label: string) {
  const steps = await page.getByTestId('timeline-step').allInnerTexts()
  const answer = await page.getByTestId('answer-text').last().innerText()
  console.log(`\n── ${label}`)
  for (const s of steps) console.log('   ' + s.replace(/\n/g, '  '))
  console.log(`   answer: ${answer.replace(/\s+/g, ' ').slice(0, 160)}`)
}

test.beforeEach(async ({ page }) => {
  await page.goto('/')
})

test('agent answers with routing, relevance, grounding and a verified stream', async ({ page }) => {
  const msg = await ask(page, 'my storage box keeps spitting out EIO5, what is going on?')
  await expect(msg.getByTestId('tool-badge')).toHaveText('Knowledge base')
  await expect(msg.getByTestId('outcome-badge')).toHaveText('answered')
  await expect(msg.getByTestId('answer-text')).toContainText('EIO-5')

  const inspector = page.getByTestId('inspector')
  await expect(inspector.getByTestId('relevance-grade').first()).toBeVisible()
  await expect(inspector.getByTestId('grounding-badge').first()).toHaveText('grounded')
  await expect(inspector.getByTestId('seq-gaps')).toHaveText('0')
  await expect(inspector.getByTestId('streams-verified')).toHaveText('1 / 0')
  expect(await page.getByTestId('timeline-step').count()).toBeGreaterThanOrEqual(6)
  await logTimeline(page, 'UI turn 1: EIO5 question')
  await page.screenshot({ path: 'e2e-artifacts/agent-answer.png' })
})

test('follow-up uses conversation memory and the thread resumes after reload', async ({ page }) => {
  await ask(page, 'my storage box keeps spitting out EIO5, what is going on?')
  const follow = await ask(page, 'and which firmware version is affected?')
  await expect(follow.getByTestId('answer-text')).toContainText('4.2.1')
  await expect(page.getByTestId('timeline-step').first()).toContainText('standalone')
  const thread = await page.getByTestId('thread-id').innerText()
  await logTimeline(page, 'UI turn 2: follow-up resolved from memory')

  await page.reload()
  await expect(page.getByTestId('history')).toContainText('Restored conversation (2 earlier turns)')
  await expect(page.getByTestId('thread-id')).toHaveText(thread)
  const third = await ask(page, 'show me recent errors on storage-07')
  await expect(third.getByTestId('tool-badge')).toHaveText('Live logs')
  await expect(third.getByTestId('answer-text')).toContainText('EIO-5 timeout')
  await logTimeline(page, 'UI turn 3 (after reload, same thread): live logs')
})

test('out-of-scope question self-corrects twice, then abstains', async ({ page }) => {
  const msg = await ask(page, 'how do I rotate the kafka broker TLS keystore?')
  await expect(msg.getByTestId('self-corrected-badge')).toHaveText('self-corrected ×2')
  await expect(msg.getByTestId('outcome-badge')).toHaveText('insufficient context')
  await expect(msg.getByTestId('answer-text')).toContainText("I couldn't find")
  await expect(page.getByTestId('relevance-grade')).toHaveCount(3)
  await expect(page.getByTestId('rewrites').locator('li')).toHaveCount(2)
  await expect(page.getByTestId('retries')).toHaveText('1 / 2')
  await logTimeline(page, 'UI turn: self-correction loop -> abstain')
  await page.screenshot({ path: 'e2e-artifacts/agent-self-correction.png' })
})

test('calculator routing', async ({ page }) => {
  const msg = await ask(page, "if the queue depth is 64 what's 64/2?")
  await expect(msg.getByTestId('tool-badge')).toHaveText('Calculator')
  await expect(msg.getByTestId('answer-text')).toHaveText('64/2 = 32')
})

test('stream mode answers over /ws/stream', async ({ page }) => {
  await page.getByTestId('mode-stream').click()
  const msg = await ask(page, 'PSU redundancy lost')
  await expect(msg.getByTestId('answer-text')).toContainText('PSU Redundancy Alerts')
  await expect(page.getByTestId('thread-id')).toHaveCount(0)
})

test('new conversation starts a fresh thread', async ({ page }) => {
  await ask(page, 'what does EIO-5 mean?')
  const first = await page.getByTestId('thread-id').innerText()
  await page.getByTestId('new-conversation').click()
  await expect(page.getByTestId('connection-status')).toHaveText('Connected')
  await expect(page.getByTestId('thread-id')).not.toHaveText(first)
  await expect(page.getByTestId('assistant-message')).toHaveCount(0)
})
