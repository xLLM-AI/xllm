// Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
import {
	buildDockerRunCommand,
	cpuArchitectures,
	hardwareConfigs,
	imageBuilds,
	type DocsLocale,
	type HardwareConfig,
} from './hardware.ts';

export function getQuickStart(locale: DocsLocale, configs: readonly HardwareConfig[] = hardwareConfigs) {
	const text = locale === 'zh' ? {
		preferences: '安装选项', hardware: '硬件平台', cpu: 'CPU 架构', build: '镜像类型',
		commandTitle: '启动容器',
		releaseBefore: '此镜像已包含编译好的 xllm 二进制文件。可跳过编译步骤，在容器内',
		releaseLink: '启动 xLLM', releaseAfter: '。',
		devBefore: '此镜像提供开发环境。请在容器内继续', devLink: '编译 xllm', devAfter: '。',
		buildAnchor: '#编译xllm', guide: '配置指南', ready: 'Docker 命令已就绪', unavailable: '未列出公开镜像',
		noImage: (hardware: string, cpu: string, build: string) => `此处未列出适用于 ${hardware} / ${cpu} 的 ${build} 镜像。请选择其他 CPU 架构或镜像类型，或查看硬件配置指南。`,
	} : {
		preferences: 'Installation preferences', hardware: 'Hardware', cpu: 'CPU', build: 'Build',
		commandTitle: 'Start container',
		releaseBefore: 'This image includes a pre-built xllm binary. Skip the build step and ',
		releaseLink: 'launch xLLM', releaseAfter: ' inside the container.',
		devBefore: 'This is a development environment. Continue with ', devLink: 'Build xllm', devAfter: ' inside the container.',
		buildAnchor: '#build-xllm', guide: 'setup guide', ready: 'Docker commands ready', unavailable: 'No public image listed',
		noImage: (hardware: string, cpu: string, build: string) => `No ${build.toLowerCase()} image is listed here for ${hardware} / ${cpu}. Choose another CPU or build, or consult the hardware guide.`,
	};
	const hardware = configs.map((config) => ({ ...config, label: config.labels[locale] }));
	const groups = [
		{ name: 'hardware', label: text.hardware, options: hardware, selected: hardware[0].id },
		{ name: 'cpu', label: text.cpu, options: cpuArchitectures, selected: 'x86' },
		{ name: 'build', label: text.build, options: imageBuilds, selected: 'release' },
	];
	const combinations = hardware.flatMap((platform) => cpuArchitectures.flatMap((cpu) => imageBuilds.map((build) => {
		const image = platform.images[cpu.id]?.[build.id];
		const command = image ? buildDockerRunCommand(platform.docker, image) : '';
		const notice = image ? '' : platform.unavailableMessage?.[locale] ?? text.noImage(platform.label, cpu.label, build.label);
		return { platform, cpu, build, image, command, notice, key: `${platform.id}-${cpu.id}-${build.id}` };
	})));
	return { text, groups, combinations };
}

export function expandQuickStartMarkdown(markdown: string, locale: DocsLocale, configs: readonly HardwareConfig[] = hardwareConfigs): string {
	return markdown.replace(/<div data-quick-start>[\s\S]*?<\/div>/, () => {
		const { text, combinations } = getQuickStart(locale, configs);
		return combinations.map(({ platform, cpu, build, image, command, notice }) => {
			const heading = `### ${platform.label} / ${cpu.label} / ${build.label}`;
			const guide = `[${platform.label} ${text.guide}](/${locale}/hardware/${platform.guide}/)`;
			if (!image) {
				return [heading, notice, guide].join('\n\n');
			}
			const nextStep = build.id === 'release'
				? `${text.releaseBefore}[${text.releaseLink}](/${locale}/getting_started/launch_xllm/)${text.releaseAfter}`
				: `${text.devBefore}[${text.devLink}](${text.buildAnchor})${text.devAfter}`;
			return [heading, `\`\`\`bash\n${command}\n\`\`\``, nextStep, guide].join('\n\n');
		}).join('\n\n');
	});
}
