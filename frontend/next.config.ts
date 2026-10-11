import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Minimal runtime bundle for the Docker image (see frontend/Dockerfile).
  output: "standalone",
};

export default nextConfig;
