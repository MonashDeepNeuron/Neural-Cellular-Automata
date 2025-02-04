import type { Config } from 'tailwindcss';

export default {
	content: ['./pages/**/*.{js,ts,jsx,tsx,mdx}', './components/**/*.{js,ts,jsx,tsx,mdx}', './app/**/*.{js,ts,jsx,tsx,mdx}'],
	theme: {
		extend: {
			colors: {
				background: 'var(--background)',
				foreground: 'var(--foreground)',
				purple: '#7065f9',
				'dark-purple': '#5a4ce9'
			}
		}
	},
	plugins: []
} satisfies Config;
