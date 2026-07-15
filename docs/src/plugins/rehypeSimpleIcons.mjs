const simpleIconDefinitions = {
	':simple-github:': {
		className: 'simple-icon-github',
		path: 'M12 .3a12 12 0 0 0-3.8 23.38c.6.12.83-.26.83-.57L9 21.07c-3.34.72-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76-1.08-.74.09-.73.09-.73 1.2.09 1.83 1.24 1.83 1.24 1.08 1.83 2.81 1.3 3.5 1 .1-.78.42-1.31.76-1.61-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.14-.3-.54-1.52.1-3.18 0 0 1-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.28-1.55 3.29-1.23 3.29-1.23.64 1.66.24 2.88.12 3.18a4.65 4.65 0 0 1 1.23 3.22c0 4.61-2.8 5.63-5.48 5.92.42.36.81 1.1.81 2.22l-.01 3.29c0 .31.2.69.82.57A12 12 0 0 0 12 .3Z',
	},
};

const simpleIconPattern = new RegExp(
	`(${Object.keys(simpleIconDefinitions)
		.map((name) => name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
		.join('|')})`,
	'g'
);
const ignoredTags = new Set(['code', 'pre', 'script', 'style']);

export default function rehypeSimpleIcons() {
	return function transform(tree) {
		replaceSimpleIconText(tree);
	};
}

function replaceSimpleIconText(parent) {
	if (!Array.isArray(parent.children)) return;
	if (parent.type === 'element' && ignoredTags.has(parent.tagName)) return;

	for (let index = 0; index < parent.children.length; index += 1) {
		const child = parent.children[index];
		if (child.type === 'text' && simpleIconPattern.test(child.value)) {
			simpleIconPattern.lastIndex = 0;
			const replacementNodes = createReplacementNodes(child.value);
			parent.children.splice(index, 1, ...replacementNodes);
			index += replacementNodes.length - 1;
			continue;
		}

		simpleIconPattern.lastIndex = 0;
		replaceSimpleIconText(child);
	}
}

function createReplacementNodes(value) {
	const nodes = [];
	let lastIndex = 0;
	let match;

	simpleIconPattern.lastIndex = 0;
	while ((match = simpleIconPattern.exec(value)) !== null) {
		if (match.index > lastIndex) {
			nodes.push({ type: 'text', value: value.slice(lastIndex, match.index) });
		}

		nodes.push(createSimpleIconNode(match[0]));
		lastIndex = match.index + match[0].length;
	}

	if (lastIndex < value.length) {
		nodes.push({ type: 'text', value: value.slice(lastIndex) });
	}

	return nodes;
}

function createSimpleIconNode(name) {
	const icon = simpleIconDefinitions[name];
	return {
		type: 'element',
		tagName: 'svg',
		properties: {
			ariaHidden: 'true',
			className: ['simple-icon', icon.className],
			fill: 'currentColor',
			focusable: 'false',
			height: '1em',
			viewBox: '0 0 24 24',
			width: '1em',
		},
		children: [
			{
				type: 'element',
				tagName: 'path',
				properties: { d: icon.path },
				children: [],
			},
		],
	};
}
