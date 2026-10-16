// Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { test } from 'node:test';
import {
	buildDockerRunCommand,
	cpuArchitectures,
	hardwareConfigs,
	imageBuilds,
	type HardwareConfig,
} from '../src/data/hardware.ts';
import { expandQuickStartMarkdown, getQuickStart } from '../src/data/quickStart.ts';

const marker = '<div data-quick-start>\n\nEnable JavaScript\n\n</div>';
const combinationCount = hardwareConfigs.length * cpuArchitectures.length * imageBuilds.length;
const imageCount = hardwareConfigs.reduce((total, { images }) => total
	+ Object.values(images).reduce((count, builds) => count + Object.values(builds).filter(Boolean).length, 0), 0);

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
		assert.equal((result.match(/^### /gm) || []).length, combinationCount);
		assert.equal((result.match(/^docker run -it/gm) || []).length, imageCount);
		assert.ok(!result.includes('docker pull'));
		assert.ok(result.includes('git submodule update --init --recursive'));
		assert.ok(result.includes('python setup.py build'));
		assert.ok(result.includes('`docker start -ai <container-name>`'));
	});

	test(`${locale}: every available selector command is exported exactly once`, () => {
		const result = expandQuickStartMarkdown(marker, locale);
		const blocks = Array.from(result.matchAll(/```bash\n([\s\S]*?)\n```/g), (match) => match[1]);
		const available = getQuickStart(locale).combinations.filter(({ image }) => image);
		assert.equal(blocks.length, imageCount);
		assert.deepEqual(blocks, available.map(({ command }) => command));
		assert.ok(blocks.every((command) => command.includes('-v "$HOME:$HOME"') && command.endsWith('/bin/bash')));
	});

	test(`${locale}: unavailable combinations export guidance instead of commands`, () => {
		const sections = expandQuickStartMarkdown(marker, locale).split(/(?=^### )/m);
		const { text, combinations } = getQuickStart(locale);
		for (const { platform, cpu, build, notice } of combinations.filter(({ image }) => !image)) {
			const heading = `### ${platform.label} / ${cpu.label} / ${build.label}\n`;
			const section = sections.find((value) => value.startsWith(heading));
			assert.ok(section, heading);
			assert.ok(!section.includes('```') && !section.includes('docker run'));
			assert.equal(notice, platform.unavailableMessage?.[locale] ?? text.noImage(platform.label, cpu.label, build.label));
			assert.ok(section.includes(notice));
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

	test(`${locale}: new hardware only needs a configuration entry, including release-only images`, () => {
		const added: HardwareConfig = {
			id: 'example', labels: { en: 'Example accelerator', zh: '示例加速卡' }, guide: 'example',
			docker: { container: 'xllm-example', options: ['--device=/dev/example0'] },
			images: { arm: { release: 'registry.example.com/xllm:release-arm' } },
		};
		const configs = [...hardwareConfigs, added];
		const { groups, combinations } = getQuickStart(locale, configs);
		assert.ok(groups[0].options.some(({ id, label }) => id === added.id && label === added.labels[locale]));
		assert.equal(combinations.length, configs.length * cpuArchitectures.length * imageBuilds.length);
		const available = combinations.find(({ key }) => key === 'example-arm-release')!;
		assert.ok(available.command.includes('--device=/dev/example0'));
		assert.ok(available.command.includes('--name xllm-example'));
		assert.ok(available.command.includes(added.images.arm!.release!));
		assert.equal(available.notice, '');
		const missing = combinations.find(({ key }) => key === 'example-arm-dev')!;
		assert.equal(missing.command, '');
		assert.ok(missing.notice.includes(added.labels[locale]));
		const markdown = expandQuickStartMarkdown(marker, locale, configs);
		assert.ok(markdown.includes(available.command));
		assert.ok(markdown.includes(missing.notice));
		assert.ok(markdown.includes(`](/${locale}/hardware/example/)`));
	});

	test(`${locale}: custom unavailable notices work without hardware-specific branches`, () => {
		const added: HardwareConfig = {
			id: 'private', labels: { en: 'Private accelerator', zh: '私有加速卡' }, guide: 'private',
			docker: { container: 'xllm-private', options: [] }, images: {},
			unavailableMessage: { en: 'Contact the vendor for an image.', zh: '请联系厂商获取镜像。' },
		};
		const { combinations } = getQuickStart(locale, [added]);
		assert.ok(combinations.every(({ command, notice }) => command === '' && notice === added.unavailableMessage![locale]));
		assert.ok(expandQuickStartMarkdown(marker, locale, [added]).includes(added.unavailableMessage![locale]));
	});
}

test('registered hardware has unique identifiers, localized labels, and existing guides', () => {
	assert.ok(hardwareConfigs.length > 0);
	assert.equal(new Set(hardwareConfigs.map(({ id }) => id)).size, hardwareConfigs.length);
	for (const hardware of hardwareConfigs) {
		assert.match(hardware.id, /^[a-z][a-z0-9-]*$/);
		assert.match(hardware.docker.container, /^[a-zA-Z0-9][a-zA-Z0-9_.-]*$/);
		for (const locale of ['en', 'zh'] as const) {
			assert.ok(hardware.labels[locale].trim());
			assert.ok(existsSync(new URL(`../src/content/docs/${locale}/hardware/${hardware.guide}.md`, import.meta.url)));
		}
	}
	const { groups, combinations } = getQuickStart('en');
	assert.deepEqual(groups[0].options.map(({ id }) => id), hardwareConfigs.map(({ id }) => id));
	assert.equal(combinations.length, combinationCount);
	assert.equal(new Set(combinations.map(({ key }) => key)).size, combinationCount);
});

test('Docker command builder preserves common arguments and hardware-specific options', () => {
	const command = buildDockerRunCommand({ container: 'xllm-example', options: ['--device=/dev/example0', '--ipc=host'] }, 'registry.example.com/xllm:dev');
	assert.equal(command, [
		'docker run -it \\',
		'  --name xllm-example \\',
		'  --device=/dev/example0 \\',
		'  --ipc=host \\',
		'  -v "$HOME:$HOME" \\',
		'  -w "$HOME" \\',
		'  registry.example.com/xllm:dev \\',
		'  /bin/bash',
	].join('\n'));
});

test('non-selector Markdown is unchanged and expansion is idempotent', () => {
	const plain = '# Example\n\n```bash\necho "$HOME"\n```\n';
	assert.equal(expandQuickStartMarkdown(plain, 'en'), plain);
	const expanded = expandQuickStartMarkdown(marker, 'en');
	assert.equal(expandQuickStartMarkdown(expanded, 'en'), expanded);
});

test('English and Chinese selectors generate identical commands without modifying configuration', () => {
	const original = JSON.stringify(hardwareConfigs);
	const commands = (locale: 'en' | 'zh') => getQuickStart(locale).combinations.map(({ key, command }) => ({ key, command }));
	assert.deepEqual(commands('en'), commands('zh'));
	assert.equal(JSON.stringify(hardwareConfigs), original);
});
