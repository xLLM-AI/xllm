// Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.

export type DocsLocale = 'en' | 'zh';
export const cpuArchitectures = [{ id: 'x86', label: 'x86_64' }, { id: 'arm', label: 'ARM64' }] as const;
export const imageBuilds = [{ id: 'release', label: 'Release' }, { id: 'dev', label: 'Dev' }] as const;
type CpuArchitecture = typeof cpuArchitectures[number]['id'];
type ImageBuild = typeof imageBuilds[number]['id'];

export interface DockerRunConfig {
	container: string;
	options: readonly string[];
}

export interface HardwareConfig {
	id: string;
	labels: Record<DocsLocale, string>;
	guide: string;
	docker: DockerRunConfig;
	images: Partial<Record<CpuArchitecture, Partial<Record<ImageBuild, string>>>>;
	unavailableMessage?: Record<DocsLocale, string>;
}

const quay = 'quay.io/jd_xllm/xllm-ai:';
const npuDocker: DockerRunConfig = {
	container: 'xllm-npu',
	options: [
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
	],
};

export const hardwareConfigs: readonly HardwareConfig[] = [
	{
		id: 'a2', labels: { en: 'Ascend A2', zh: '昇腾 A2' }, guide: 'ascend_npu', docker: npuDocker,
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
		id: 'a3', labels: { en: 'Ascend A3', zh: '昇腾 A3' }, guide: 'ascend_npu', docker: npuDocker,
		images: {
			arm: {
				release: `${quay}xllm-0.10.0-release-hc-rc2-arm`,
				dev: `${quay}xllm-dev-a3-arm-cann9-20260911`,
			},
		},
	},
	{
		id: 'cuda', labels: { en: 'NVIDIA GPU', zh: 'NVIDIA GPU' }, guide: 'nvidia_gpu',
		docker: {
			container: 'xllm-cuda',
			options: ['--gpus all', '--privileged', '--shm-size=128g', '--ipc=host', '--network=host', '--pid=host'],
		},
		images: {
			x86: {
				release: `${quay}xllm-0.8.0-release-cuda-x86`,
				dev: `${quay}xllm-dev-cuda-x86`,
			},
		},
	},
	{
		id: 'mlu', labels: { en: 'Cambricon MLU', zh: '寒武纪 MLU' }, guide: 'cambricon_mlu',
		docker: { container: 'xllm-mlu', options: [] }, images: {},
		unavailableMessage: {
			en: 'A public Cambricon MLU image is not provided. Obtain a compatible development image from your vendor and follow the hardware guide.',
			zh: '暂不提供公开的寒武纪 MLU 镜像。请向厂商获取兼容的开发镜像，并按照硬件配置指南操作。',
		},
	},
	{
		id: 'dcu', labels: { en: 'Hygon DCU', zh: '海光 DCU' }, guide: 'dcu',
		docker: {
			container: 'xllm-dcu',
			options: [
				'--ipc=host', '-u 0', '--privileged', '--network=host', '--shm-size=256g',
				'--device=/dev/kfd', '--device=/dev/dri', '--device=/dev/mkfd',
				'--security-opt seccomp=unconfined', '--group-add video', '-v /opt/hyhal:/opt/hyhal',
			],
		},
		images: { x86: { dev: 'harbor.sourcefind.cn:5443/dcu/admin/base/custom:xllm-dev-dcu-x86-20260617' } },
	},
	{
		id: 'maca', labels: { en: 'MetaX MACA', zh: '沐曦 MACA' }, guide: 'metax_maca',
		docker: {
			container: 'xllm-maca',
			options: [
				'--ipc=host', '-u 0', '--network=host', '--privileged', '--shm-size=100g',
				'--device=/dev/mxcd', '--device=/dev/dri', '--device=/dev/infiniband',
				'--security-opt seccomp=unconfined', '--security-opt apparmor=unconfined',
				'--group-add video', '--ulimit memlock=-1', '-v /opt/maca:/opt/maca',
			],
		},
		images: { x86: { dev: 'pub-registry1.metax-tech.com/dev-m01421/xllm-maca3.7.1.9:v1' } },
	},
	{
		id: 'musa', labels: { en: 'Mthreads MUSA', zh: '摩尔线程 MUSA' }, guide: 'musa',
		docker: {
			container: 'xllm-musa',
			options: [
				'--ipc=host', '--network=host', '--privileged', '--shm-size=128g',
				'--device=/dev/mtgpu0', '--device=/dev/dri', '--group-add video', '--ulimit memlock=-1',
			],
		},
		images: { x86: { dev: 'registry.mthreads.com/presale/devtech/xllm:0710' } },
	},
];

export function buildDockerRunCommand(docker: DockerRunConfig, image: string): string {
	return [
		'docker run -it', `--name ${docker.container}`, ...docker.options,
		'-v "$HOME:$HOME"', '-w "$HOME"', image, '/bin/bash',
	].join(' \\\n  ');
}
