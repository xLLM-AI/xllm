// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightThemeRapide from 'starlight-theme-rapide';
import rehypeSimpleIcons from './src/plugins/rehypeSimpleIcons.mjs';

// https://astro.build/config
export default defineConfig({
	markdown: {
		rehypePlugins: [rehypeSimpleIcons],
	},
	integrations: [
		starlight({
			plugins: [starlightThemeRapide()],
			customCss: ['./src/styles/theme.css'],
			lastUpdated: true,
			components: {
				Head: './src/components/Head.astro',
				Header: './src/components/Header.astro',
				PageTitle: './src/components/PageTitle.astro',
				SiteTitle: './src/components/SiteTitle.astro',
			},
			title: {
				en: 'xLLM',
				'zh-CN': 'xLLM',
			},
			locales: {
				en: {
					label: 'EN',
					lang: 'en',
				},
				zh: {
					label: '中',
					lang: 'zh-CN',
				},
			},
			defaultLocale: 'en',
			logo: {
				src: './src/assets/logo_with_llm.png',
				alt: 'xLLM',
				replacesTitle: true,
			},
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/xLLM-AI/xllm' }],
			sidebar: [
				{ label: 'Home', translations: { 'zh-CN': '主页' }, slug: '' },
				{
					label: 'Getting Started',
					translations: { 'zh-CN': '开始使用' },
					items: [
						'getting_started/quick_start',
						'getting_started/launch_xllm',
						'getting_started/multi_machine',
						'getting_started/online_service',
						'getting_started/offline_service',
						{
							label: 'Supported Models',
							translations: { 'zh-CN': '模型支持列表' },
							slug: 'supported_models',
						},
					],
				},
				{
					label: 'Hardware',
					translations: { 'zh-CN': '硬件' },
					items: [
						{
							label: 'Hardware Platforms',
							translations: { 'zh-CN': '硬件平台' },
							slug: 'hardware/overview',
						},
						{
							label: 'NVIDIA GPU',
							translations: { 'zh-CN': 'NVIDIA GPU' },
							slug: 'hardware/nvidia_gpu',
						},
						{
							label: 'Ascend NPU',
							translations: { 'zh-CN': '昇腾 NPU' },
							slug: 'hardware/ascend_npu',
						},
						{
							label: 'Cambricon MLU',
							translations: { 'zh-CN': '寒武纪 MLU' },
							slug: 'hardware/cambricon_mlu',
						},
						{
							label: 'Hygon DCU',
							translations: { 'zh-CN': '海光 DCU' },
							slug: 'hardware/dcu',
						},
						{
							label: 'MetaX MACA',
							translations: { 'zh-CN': '沐曦 MACA' },
							slug: 'hardware/metax_maca',
						},
						{
							label: 'Mthreads MUSA',
							translations: { 'zh-CN': '摩尔线程 MUSA' },
							slug: 'hardware/musa',
						},
					],
				},
				{
					label: 'User Guide',
					translations: { 'zh-CN': '用户指南' },
					items: [
						{
							label: 'Advanced Features',
							translations: { 'zh-CN': '高级功能' },
							autogenerate: { directory: 'features' },
						},
					],
				},
				{
					label: 'CookBook',
					translations: { 'zh-CN': '实践指南' },
					items: [
						{
							label: 'Autoregressive Models',
							translations: { 'zh-CN': '自回归模型' },
							collapsed: false,
							items: [
								{
									label: 'Qwen',
									translations: { 'zh-CN': 'Qwen' },
									collapsed: true,
									items: [
										{
											label: 'Qwen3.5',
											translations: { 'zh-CN': 'Qwen3.5' },
											slug: 'cookbook/autoregressive_models/qwen/qwen3_5',
										},
										{
											label: 'Qwen3',
											translations: { 'zh-CN': 'Qwen3' },
											slug: 'cookbook/autoregressive_models/qwen/qwen3',
										},
										{
											label: 'Qwen3-Next',
											translations: { 'zh-CN': 'Qwen3-Next' },
											slug: 'cookbook/autoregressive_models/qwen/qwen3_next',
										},
										{
											label: 'Qwen3-VL',
											translations: { 'zh-CN': 'Qwen3-VL' },
											slug: 'cookbook/autoregressive_models/qwen/qwen3_vl',
										},
										{
											label: 'Qwen2.5-VL',
											translations: { 'zh-CN': 'Qwen2.5-VL' },
											slug: 'cookbook/autoregressive_models/qwen/qwen2_5_vl',
										},
									],
								},
								{
									label: 'DeepSeek',
									translations: { 'zh-CN': 'DeepSeek' },
									collapsed: true,
									items: [
										{
											label: 'DeepSeek-V4',
											translations: { 'zh-CN': 'DeepSeek-V4' },
											slug: 'cookbook/autoregressive_models/deepseek/deepseek_v4',
										},
										{
											label: 'DeepSeek-V3.2',
											translations: { 'zh-CN': 'DeepSeek-V3.2' },
											slug: 'cookbook/autoregressive_models/deepseek/deepseek_v3_2',
										},
										{
											label: 'DeepSeek-V3.1',
											translations: { 'zh-CN': 'DeepSeek-V3.1' },
											slug: 'cookbook/autoregressive_models/deepseek/deepseek_v3_1',
										},
										{
											label: 'DeepSeek-V3',
											translations: { 'zh-CN': 'DeepSeek-V3' },
											slug: 'cookbook/autoregressive_models/deepseek/deepseek_v3',
										},
										{
											label: 'DeepSeek-R1',
											translations: { 'zh-CN': 'DeepSeek-R1' },
											slug: 'cookbook/autoregressive_models/deepseek/deepseek_r1',
										},
									],
								},
								{
									label: 'GLM',
									translations: { 'zh-CN': 'GLM' },
									collapsed: true,
									items: [
										{
											label: 'GLM-5.2',
											translations: { 'zh-CN': 'GLM-5.2' },
											slug: 'cookbook/autoregressive_models/glm/glm_5',
										},
										{
											label: 'GLM-5.1',
											translations: { 'zh-CN': 'GLM-5.1' },
											slug: 'cookbook/autoregressive_models/glm/glm_5',
										},
										{
											label: 'GLM-5',
											translations: { 'zh-CN': 'GLM-5' },
											slug: 'cookbook/autoregressive_models/glm/glm_5',
										},
										{
											label: 'GLM-4.7',
											translations: { 'zh-CN': 'GLM-4.7' },
											slug: 'cookbook/autoregressive_models/glm/glm_4_7',
										},
										{
											label: 'GLM-4.7-Flash',
											translations: { 'zh-CN': 'GLM-4.7-Flash' },
											slug: 'cookbook/autoregressive_models/glm/glm_4_7_flash',
										},
										{
											label: 'GLM-4.6',
											translations: { 'zh-CN': 'GLM-4.6' },
											slug: 'cookbook/autoregressive_models/glm/glm_4_6',
										},
										{
											label: 'GLM-4.6V',
											translations: { 'zh-CN': 'GLM-4.6V' },
											slug: 'cookbook/autoregressive_models/glm/glm_4_6v',
										},
										{
											label: 'GLM-4.5',
											translations: { 'zh-CN': 'GLM-4.5' },
											slug: 'cookbook/autoregressive_models/glm/glm_4_5',
										},
										{
											label: 'GLM-4.5V',
											translations: { 'zh-CN': 'GLM-4.5V' },
											slug: 'cookbook/autoregressive_models/glm/glm_4_5v',
										},
									],
								},
								{
									label: 'Kimi',
									translations: { 'zh-CN': 'Kimi' },
									collapsed: true,
									items: [
										{
											label: 'Kimi2',
											translations: { 'zh-CN': 'Kimi2' },
											slug: 'cookbook/autoregressive_models/kimi/kimi2',
										},
										{
											label: 'Kimi-K2.5 / Kimi-K2.6',
											translations: { 'zh-CN': 'Kimi-K2.5 / Kimi-K2.6' },
											slug: 'cookbook/autoregressive_models/kimi/kimi2_5',
										},
									],
								},
								{
									label: 'MinMax',
									translations: { 'zh-CN': 'MinMax' },
									collapsed: true,
									items: [
										{
											label: 'MiniMax-M2.7',
											translations: { 'zh-CN': 'MiniMax-M2.7' },
											slug: 'cookbook/autoregressive_models/minmax/minmax_m2_7',
										},
									],
								},
							],
						},
						{
							label: 'Diffusion Models',
							translations: { 'zh-CN': '扩散模型' },
							collapsed: false,
							items: [
								{
									label: 'Flux',
									translations: { 'zh-CN': 'Flux' },
									collapsed: true,
									items: [
										{
											label: 'Flux',
											translations: { 'zh-CN': 'Flux' },
											slug: 'cookbook/diffusion_models/flux/flux',
										},
										{
											label: 'Flux2',
											translations: { 'zh-CN': 'Flux2' },
											slug: 'cookbook/diffusion_models/flux/flux2',
										},
									],
								},
								{
									label: 'Wan',
									translations: { 'zh-CN': 'Wan' },
									collapsed: true,
									items: [
										{
											label: 'Wan2.1',
											translations: { 'zh-CN': 'Wan2.1' },
											slug: 'cookbook/diffusion_models/wan/wan2_1',
										},
									],
								},
								{
									label: 'Qwen-Image',
									translations: { 'zh-CN': 'Qwen-Image' },
									collapsed: true,
									items: [
										{
											label: 'Qwen-Image',
											translations: { 'zh-CN': 'Qwen-Image' },
											slug: 'cookbook/diffusion_models/qwen_image/qwen_image',
										},
									],
								},
							],
						},
					],
				},
				{
					label: 'Developer Guide',
					translations: { 'zh-CN': '开发者指南' },
					items: [
						{
							label: 'Development',
							translations: { 'zh-CN': '开发' },
							autogenerate: { directory: 'dev_guide' },
						},
						{
							label: 'Design',
							translations: { 'zh-CN': '设计文档' },
							autogenerate: { directory: 'design' },
						},
					],
				},
				{ label: 'CLI Reference', translations: { 'zh-CN': 'CLI 参考' }, slug: 'cli_reference' },
			],
		}),
	],
});
