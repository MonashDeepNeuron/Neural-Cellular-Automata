import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
	output: 'export',
};

module.exports = {
    images: {
        unoptimized: true
    }
}

export default nextConfig;
