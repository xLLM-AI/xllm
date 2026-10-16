// Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { expandQuickStartMarkdown, getQuickStart } from '../src/data/quickStart.ts';

const marker = '<div data-quick-start>\n\nEnable JavaScript\n\n</div>';

for (const locale of ['en', 'zh'] as const) {
	test(`${locale}: expand the actual Quick Start source without changing surrounding content`, () => {
		const source = readFileSync(new URL(`../src/content/docs/${locale}/getting_started/quick_start.md`, import.meta.url), 'utf-8');
		const placeholder = source.match(/<div data-quick-start>[\s\S]*?<\/div>/);
		assert.ok(placeholder);
		const [before, after] = source.split(placeholder[0]);
		const result = expandQuickStartMarkdown(source, locale);
		assert.ok(result.startsWith(before));
		assert.ok(result.endsWith(after));
		assert.ok(!result.includes('data-quick-start'));
		assert.ok(!result.includes('Enable JavaScript') && !result.includes('请启用 JavaScript'));
		assert.equal((result.match(/^### /gm) || []).length, 28);
		assert.equal((result.match(/^docker run -it/gm) || []).length, 11);
		assert.ok(!result.includes('docker pull'));
		assert.ok(result.includes('git submodule update --init --recursive'));
		assert.ok(result.includes('python setup.py build'));
	});

	test(`${locale}: every available selector command is exported exactly once`, () => {
		const result = expandQuickStartMarkdown(marker, locale);
		const blocks = Array.from(result.matchAll(/```bash\n([\s\S]*?)\n```/g), (match) => match[1]);
		const available = getQuickStart(locale).combinations.filter(({ image }) => image);
		assert.equal(blocks.length, 11);
		assert.deepEqual(blocks, available.map(({ command }) => command));
		for (const { image } of available) {
			assert.equal(result.split(image!).length - 1, 1);
		}
		assert.ok(blocks.every((command) => command.includes('-v "$HOME:$HOME"') && command.endsWith('/bin/bash')));
	});

	test(`${locale}: unavailable combinations export guidance instead of commands`, () => {
		const sections = expandQuickStartMarkdown(marker, locale).split(/(?=^### )/m);
		const { text, combinations } = getQuickStart(locale);
		for (const { platform, cpu, build } of combinations.filter(({ image }) => !image)) {
			const heading = `### ${platform.label} / ${cpu.label} / ${build.label}\n`;
			const section = sections.find((value) => value.startsWith(heading));
			assert.ok(section, heading);
			assert.ok(!section.includes('```') && !section.includes('docker run'));
			assert.ok(section.includes(platform.id === 'mlu' ? text.mluUnavailable : text.noImage(platform.label, cpu.label, build.label)));
			assert.ok(section.includes(`](/${locale}/hardware/${platform.guide}/)`));
		}
	});

	test(`${locale}: copied instructions use localized build, launch, and hardware links`, () => {
		const result = expandQuickStartMarkdown(marker, locale);
		const otherLocale = locale === 'zh' ? 'en' : 'zh';
		assert.ok(!result.includes(`](/${otherLocale}/`));
		assert.ok(result.includes(`](/${locale}/getting_started/launch_xllm/)`));
		assert.ok(result.includes(locale === 'zh' ? '[编译 xllm](#编译xllm)' : '[Build xllm](#build-xllm)'));
		assert.ok(result.includes(locale === 'zh' ? '昇腾 A2 / x86_64 / Release' : 'Ascend A2 / x86_64 / Release'));
		assert.ok(result.includes(locale === 'zh' ? '此镜像已包含编译好的' : 'This image includes a pre-built'));
	});
}

test('non-selector Markdown is unchanged and expansion is idempotent', () => {
	const plain = '# Example\n\n```bash\necho "$HOME"\n```\n';
	assert.equal(expandQuickStartMarkdown(plain, 'en'), plain);
	const expanded = expandQuickStartMarkdown(marker, 'en');
	assert.equal(expandQuickStartMarkdown(expanded, 'en'), expanded);
});

test('English and Chinese selectors generate identical commands', () => {
	const commands = (locale: 'en' | 'zh') => getQuickStart(locale).combinations.map(({ key, command }) => ({ key, command }));
	assert.deepEqual(commands('en'), commands('zh'));
});
