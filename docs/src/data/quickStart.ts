// Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.

interface Hardware {
	id: string;
	label: string;
	guide: string;
	container: string;
	options: string[];
	images: Partial<Record<'x86' | 'arm', { release?: string; dev: string }>>;
}

export function getQuickStart(locale: 'en' | 'zh') {
	const text = locale === 'zh' ? {
		preferences: '安装选项', hardware: '硬件平台', cpu: 'CPU 架构', build: '镜像类型',
		platforms: { a2: '昇腾 A2', a3: '昇腾 A3', cuda: 'NVIDIA GPU', mlu: '寒武纪 MLU', dcu: '海光 DCU', maca: '沐曦 MACA', musa: '摩尔线程 MUSA' },
		commandTitle: '启动容器',
		releaseBefore: '此镜像已包含编译好的 xllm 二进制文件。可跳过编译步骤，在容器内',
		releaseLink: '启动 xLLM', releaseAfter: '。',
		devBefore: '此镜像提供开发环境。请在容器内继续', devLink: '编译 xllm', devAfter: '。',
		buildAnchor: '#编译xllm', guide: '配置指南', ready: 'Docker 命令已就绪', unavailable: '未列出公开镜像',
		mluUnavailable: '暂不提供公开的寒武纪 MLU 镜像。请向厂商获取兼容的开发镜像，并按照硬件配置指南操作。',
		noImage: (hardware: string, cpu: string, build: string) => `此处未列出适用于 ${hardware} / ${cpu} 的 ${build} 镜像。请选择其他 CPU 架构或镜像类型，或查看硬件配置指南。`,
	} : {
		preferences: 'Installation preferences', hardware: 'Hardware', cpu: 'CPU', build: 'Build',
		platforms: { a2: 'Ascend A2', a3: 'Ascend A3', cuda: 'NVIDIA GPU', mlu: 'Cambricon MLU', dcu: 'Hygon DCU', maca: 'MetaX MACA', musa: 'Mthreads MUSA' },
		commandTitle: 'Start container',
		releaseBefore: 'This image includes a pre-built xllm binary. Skip the build step and ',
		releaseLink: 'launch xLLM', releaseAfter: ' inside the container.',
		devBefore: 'This is a development environment. Continue with ', devLink: 'Build xllm', devAfter: ' inside the container.',
		buildAnchor: '#build-xllm', guide: 'setup guide', ready: 'Docker commands ready', unavailable: 'No public image listed',
		mluUnavailable: 'A public Cambricon MLU image is not provided. Obtain a compatible development image from your vendor and follow the hardware guide.',
		noImage: (hardware: string, cpu: string, build: string) => `No ${build.toLowerCase()} image is listed here for ${hardware} / ${cpu}. Choose another CPU or build, or consult the hardware guide.`,
	};
	const quay = 'quay.io/jd_xllm/xllm-ai:';
	const npuOptions = [
		'--ipc=host', '-u 0', '--privileged', '--network=host',
		'--device=/dev/davinci0', '--device=/dev/davinci_manager',
		'--device=/dev/devmm_svm', '--device=/dev/hisi_hdc',
		'-v /usr/local/Ascend/driver:/usr/local/Ascend/driver',
		'-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/',
		'-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi',
		'-v /usr/local/sbin/:/usr/local/sbin/',
		'-v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf',
		'-v /var/log/npu/slog/:/var/log/npu/slog',
		'-v /var/log/npu/profiling/:/var/log/npu/profiling',
		'-v /var/log/npu/dump/:/var/log/npu/dump',
	];
	const hardware: Hardware[] = [
		{
			id: 'a2', label: text.platforms.a2, guide: 'ascend_npu', container: 'npu', options: npuOptions,
			images: {
				x86: {
					release: `${quay}xllm-0.10.0-release-hb-rc2-x86`,
					dev: `${quay}xllm-dev-a2-x86-cann9-20260911`,
				},
				arm: {
					release: `${quay}xllm-0.10.0-release-hb-rc2-arm`,
					dev: `${quay}xllm-dev-a2-arm-cann9-20260911`,
				},
			},
		},
		{
			id: 'a3', label: text.platforms.a3, guide: 'ascend_npu', container: 'npu', options: npuOptions,
			images: {
				arm: {
					release: `${quay}xllm-0.10.0-release-hc-rc2-arm`,
					dev: `${quay}xllm-dev-a3-arm-cann9-20260911`,
				},
			},
		},
		{
			id: 'cuda', label: text.platforms.cuda, guide: 'nvidia_gpu', container: 'cuda',
			options: ['--gpus all', '--privileged', '--shm-size=128g', '--ipc=host', '--network=host', '--pid=host'],
			images: {
				x86: {
					release: `${quay}xllm-0.8.0-release-cuda-x86`,
					dev: `${quay}xllm-dev-cuda-x86`,
				},
			},
		},
		{
			id: 'mlu', label: text.platforms.mlu, guide: 'cambricon_mlu', container: 'mlu',
			options: [], images: {},
		},
		{
			id: 'dcu', label: text.platforms.dcu, guide: 'dcu', container: 'dcu',
			options: [
				'--ipc=host', '-u 0', '--privileged', '--network=host', '--shm-size=256g',
				'--device=/dev/kfd', '--device=/dev/dri', '--device=/dev/mkfd',
				'--security-opt seccomp=unconfined', '--group-add video', '-v /opt/hyhal:/opt/hyhal',
			],
			images: { x86: { dev: 'harbor.sourcefind.cn:5443/dcu/admin/base/custom:xllm-dev-dcu-x86-20260617' } },
		},
		{
			id: 'maca', label: text.platforms.maca, guide: 'metax_maca', container: 'maca',
			options: [
				'--ipc=host', '-u 0', '--network=host', '--privileged', '--shm-size=100g',
				'--device=/dev/mxcd', '--device=/dev/dri', '--device=/dev/infiniband',
				'--security-opt seccomp=unconfined', '--security-opt apparmor=unconfined',
				'--group-add video', '--ulimit memlock=-1', '-v /opt/maca:/opt/maca',
			],
			images: { x86: { dev: 'pub-registry1.metax-tech.com/dev-m01421/xllm-maca3.7.1.9:v1' } },
		},
		{
			id: 'musa', label: text.platforms.musa, guide: 'musa', container: 'musa',
			options: [
				'--ipc=host', '--network=host', '--privileged', '--shm-size=128g',
				'--device=/dev/mtgpu0', '--device=/dev/dri', '--group-add video', '--ulimit memlock=-1',
			],
			images: { x86: { dev: 'registry.mthreads.com/presale/devtech/xllm:0710' } },
		},
	];
	const cpus = [{ id: 'x86', label: 'x86_64' }, { id: 'arm', label: 'ARM64' }] as const;
	const builds = [{ id: 'release', label: 'Release' }, { id: 'dev', label: 'Dev' }] as const;
	const groups = [
		{ name: 'hardware', label: text.hardware, options: hardware, selected: 'a2' },
		{ name: 'cpu', label: text.cpu, options: cpus, selected: 'x86' },
		{ name: 'build', label: text.build, options: builds, selected: 'release' },
	];
	const combinations = hardware.flatMap((platform) => cpus.flatMap((cpu) => builds.map((build) => {
		const image = platform.images[cpu.id]?.[build.id];
		const command = image ? [
			'docker run -it', `--name xllm-${platform.container}`, ...platform.options,
			'-v "$HOME:$HOME"', '-w "$HOME"', image, '/bin/bash',
		].join(' \\\n  ') : '';
		return { platform, cpu, build, image, command, key: `${platform.id}-${cpu.id}-${build.id}` };
	})));
	return { text, groups, combinations };
}

export function expandQuickStartMarkdown(markdown: string, locale: 'en' | 'zh'): string {
	return markdown.replace(/<div data-quick-start>[\s\S]*?<\/div>/, () => {
		const { text, combinations } = getQuickStart(locale);
		return combinations.map(({ platform, cpu, build, image, command }) => {
			const heading = `### ${platform.label} / ${cpu.label} / ${build.label}`;
			const guide = `[${platform.label} ${text.guide}](/${locale}/hardware/${platform.guide}/)`;
			if (!image) {
				const notice = platform.id === 'mlu' ? text.mluUnavailable : text.noImage(platform.label, cpu.label, build.label);
				return [heading, notice, guide].join('\n\n');
			}
			const nextStep = build.id === 'release'
				? `${text.releaseBefore}[${text.releaseLink}](/${locale}/getting_started/launch_xllm/)${text.releaseAfter}`
				: `${text.devBefore}[${text.devLink}](${text.buildAnchor})${text.devAfter}`;
			return [heading, `\`\`\`bash\n${command}\n\`\`\``, nextStep, guide].join('\n\n');
		}).join('\n\n');
	});
}
