# Build from the repository root: docker build -f frontend/Dockerfile .
# ---- deps ----
FROM node:24-alpine AS deps
WORKDIR /app
RUN corepack enable
# pnpm-workspace.yaml carries the allowed-build-scripts list pnpm 10+ requires.
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
# The store is a BuildKit cache mount, so packages already downloaded survive a failed or
# interrupted build and a retry resumes instead of starting over (matters on a slow network).
RUN --mount=type=cache,id=pnpm-store,target=/pnpm/store \
    pnpm install --frozen-lockfile --store-dir /pnpm/store \
    --fetch-timeout 300000 --fetch-retries 5

# ---- build ----
FROM node:24-alpine AS build
WORKDIR /app
RUN corepack enable
COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ .
# Baked in at build time: the browser needs to know where the backend is.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_TELEMETRY_DISABLED=1
RUN pnpm build

# ---- run ----
FROM node:24-alpine AS run
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
EXPOSE 3000
ENV PORT=3000 HOSTNAME=0.0.0.0
CMD ["node", "server.js"]
