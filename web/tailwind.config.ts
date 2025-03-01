import type { Config } from 'tailwindcss';

export default {
	content: ['./pages/**/*.{js,ts,jsx,tsx,mdx}', './components/**/*.{js,ts,jsx,tsx,mdx}', './app/**/*.{js,ts,jsx,tsx,mdx}'],
	theme: {
		extend: {
			colors: {
				background: 'var(--background)',
				foreground: 'var(--foreground)',
				'purple-mdn': '#7065f9',
				'purple-mdn-dark': '#4838d6',
				'purple-mdn-lighter': '#7f90f9',
				'purple-mdn-light': '#afc0f2'
			}
		}
	},
	plugins: []
} satisfies Config;
