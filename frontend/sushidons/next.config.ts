// next.config.ts
import { NextConfig } from 'next';

const nextConfig: NextConfig = {
  reactStrictMode: false, // Temporarily disable strict mode
  // Add custom webpack configuration to handle hydration issues
  webpack: (config, { isServer }) => {
    if (!isServer) {
      // Ignore browser extension related attributes during client-side rendering
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
      };
    }
    return config;
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'www.sushidon.shop',
        pathname: '/**',
      },
    ],
  },
};

export default nextConfig;
