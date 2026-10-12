import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

const loader = docsLoader();
const homepageDescriptions = new Map([
	['en', 'xLLM documentation home'],
	['zh', 'xLLM 文档首页'],
]);

export const collections = {
	docs: defineCollection({
		loader: {
			...loader,
			load(context) {
				return loader.load({
					...context,
					parseData(props) {
						const description = homepageDescriptions.get(props.id);
						return context.parseData(description ? {
							...props,
							data: {
								...props.data,
								title: 'xLLM',
								description,
								lastUpdated: false,
								head: [{ tag: 'title', content: 'xLLM Documentation' }],
							},
						} : props);
					},
				});
			},
		},
		schema: docsSchema(),
	}),
};
