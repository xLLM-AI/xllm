// Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
import { dirname, relative, resolve } from 'node:path';

const readmeLinks = { './README.md': '/en/', './README_zh.md': '/zh/' };

export default function remarkReadmeHomepage() {
	return function transform(tree, file) {
		if (!/\/src\/content\/docs\/(en|zh)\/index\.md$/.test(file.path || '')) return;

		const pageDirectory = dirname(file.path);
		const repositoryRoot = resolve(pageDirectory, '../../../../..');
		rewriteNodes(tree, pageDirectory, repositoryRoot);
	};
}

function rewriteNodes(parent, pageDirectory, repositoryRoot) {
	if (!Array.isArray(parent.children)) return;

	parent.children = parent.children.flatMap((node) => {
		if (node.type === 'paragraph' && node.children.some(isBadgeLink)) {
			node.data ??= {};
			node.data.hProperties ??= {};
			node.data.hProperties.className = ['readme-badges'];
		}
		if (node.type === 'link' && Object.hasOwn(readmeLinks, node.url)) {
			node.url = readmeLinks[node.url];
		}
		if (node.type === 'image' && node.url.startsWith('assets/')) {
			node.url = relative(pageDirectory, resolve(repositoryRoot, node.url)).replaceAll('\\', '/');
		}
		rewriteNodes(node, pageDirectory, repositoryRoot);
		if (node.type !== 'html') return [node];

		const replacements = [];
		let offset = 0;
		for (const match of node.value.matchAll(/<img\b[^>]*\bsrc="(assets\/[^"<>]+)"[^>]*>/g)) {
			if (match.index > offset) {
				replacements.push({ type: 'html', value: node.value.slice(offset, match.index) });
			}
			const alt = /\balt="([^"]*)"/.exec(match[0])?.[1] || '';
			const style = /\bstyle="([^"]*)"/.exec(match[0])?.[1];
			const width = /\bwidth="([^"]*)"/.exec(match[0])?.[1];
			replacements.push({
				type: 'paragraph',
				children: [{
					type: 'image',
					alt,
					url: relative(pageDirectory, resolve(repositoryRoot, match[1])).replaceAll('\\', '/'),
					data: { hProperties: { style: style || (width ? `width:${width}; height:auto;` : undefined) } },
				}],
			});
			offset = match.index + match[0].length;
		}
		if (!offset) return [node];
		if (offset < node.value.length) {
			replacements.push({ type: 'html', value: node.value.slice(offset) });
		}
		return replacements;
	});
}

function isBadgeLink(node) {
	return node.type === 'link' && node.children.some((child) =>
		child.type === 'image' &&
		child.url.startsWith('https://img.shields.io/')
	);
}
